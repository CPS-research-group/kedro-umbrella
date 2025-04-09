import numpy as np

import os
import pickle
from pythonfmu3.builder import FmuBuilder
#from fmpy import read_model_description, extract
#from fmpy.fmi3 import FMU3Slave
import uuid

import logging
logger = logging.getLogger(__name__)

def get_fmu_variable_type(data):
    if data.dtype == np.float64:
        return "Float64"
    elif data.dtype == np.float32:
        return "Float32"
    elif data.dtype == np.int32:
        return "Int32"
    elif data.dtype == np.int64:
        return "Int64"
    elif data.dtype == np.uint64:
        return "UInt64"
    elif data.dtype == np.bool_:
        return "Boolean"
    elif data.dtype == np.object_ or data.dtype == np.str_:
        return "String"
    else:
        raise ValueError(f"Unsupported data type for FMU: {data.dtype}")

def generate_dimensions(shape, data_type):
    if data_type == "Float64":
        dimensions = [f"Dimension(start=f\"{1}\")"]
        if len(shape) > 1:
            dimensions.append(f"Dimension(start=f\"{shape[1]}\")")
    else:
        dimensions = [f"Dimension(start=f\"{dim}\")" for dim in shape]
    return ", ".join(dimensions)

def get_unique_variable_types(X_train, Y_train):
    types = {get_fmu_variable_type(X_train), get_fmu_variable_type(Y_train)}

    return types

def generate_guid():
    return str(uuid.uuid4())

def generate_model_description_xml(guid):
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<fmiModelDescription
    fmiVersion="3.0"
    modelName="SpringModelPredictor"
    guid="{guid}"
    ...>
    <CoSimulation
        modelIdentifier="SpringModelPredictor"
        canGetAndSetFMUState="true"
        canSerializeFMUState="true"
        ...>
        ...
    </CoSimulation>
</fmiModelDescription>
"""
    return xml_content


def generate_fmu_class(X_train, Y_train, *models):

    fmu_dir = 'data/06_models/dynamic_fmu'
    os.makedirs(fmu_dir, exist_ok=True)

    script_file_path = os.path.join(fmu_dir, 'dynamicFMU.py')
    fmu_output_path = os.path.join(fmu_dir, 'dynamicFMU.fmu')

    X_type = get_fmu_variable_type(X_train)
    Y_type = get_fmu_variable_type(Y_train)
    X_dimensions = generate_dimensions(X_train.shape, X_type)
    Y_dimensions = generate_dimensions(Y_train.shape, Y_type)
    unique_types = get_unique_variable_types(X_train, Y_train)
    variable_types_import = ", ".join(unique_types)

    # Dump the resource in the folder
    for i, model in enumerate(models):
        pickle.dump(model, open(f'{fmu_dir}/model_{i + 1}.pkl', 'wb'))

    model_loading_code = "\n        ".join(
        [f"self.model_{i + 1} = pickle.load(open('{fmu_dir}/model_{i + 1}.pkl', 'rb'))" 
        for i in range(len(models))])

    model_list = "self.model_list = [" + ", ".join([f"self.model_{i + 1}" for i in range(len(models))]) + "]"

    fmu_class_template = f"""
import numpy as np
from pythonfmu3 import Fmi3Causality, Fmi3Slave, {variable_types_import}, Fmi3Variability, Dimension
import pickle

class DynamicFMU(Fmi3Slave):
    author = "Generated"
    description = "A dynamically generated FMU"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load the resource from the FMU folder
        {model_loading_code}

        {model_list}

        self.sample_x = np.zeros((1, {X_train.shape[1]}))
        self.y_pred = np.zeros((1, {Y_train.shape[1]}))

        self.register_variable(Float64('time', causality=Fmi3Causality.independent, variability=Fmi3Variability.continuous))
        self.register_variable({X_type}("sample_x", causality=Fmi3Causality.input, dimensions=[{X_dimensions}]))
        self.register_variable({Y_type}("y_pred", causality=Fmi3Causality.output, dimensions=[{Y_dimensions}]))

    def do_step(self, current_time, step_size):
        try:
            if self.sample_x is None:
                print("sample_x not provided.")
                return False

            data = self.sample_x

            for i, model in enumerate(self.model_list):
                data = model(data)

            self.y_pred = data
            return True
        except Exception as e:
            print(f"Error during model application: {{e}}")
            return False
    """

    with open(script_file_path, 'w') as file:
        file.write(fmu_class_template)


    logger.info("FMU class template generated successfully.")

    FmuBuilder.build_FMU(
        script_file=script_file_path,
        dest=fmu_output_path,
        needs_execution_tool=False,
        can_get_and_set_fmu_state=True,
        can_serialize_fmu_state=True
    )
    
    logger.info(f"FMU exported to {fmu_output_path}")
    return script_file_path
