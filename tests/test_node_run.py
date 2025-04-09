# pylint: disable=unused-argument

import pytest

from kedro_umbrella import coder, processor, trainer
from kedro_umbrella.types import TypeCatalog, DataType, FunctionType


def one_in_one_out(arg):
    return arg

def zero_in_one_out():
    return 42

def one_in_dict_out(arg):
    return {"ret": arg}


def two_in_first_out(arg1, arg2):
    return arg1


def code_one_func():
    return one_in_one_out


def code_one_func_in1(in1):
    return one_in_one_out


def code_one_func_in2(in1, in2):
    return one_in_one_out


def train_one_func_in3(in1, in2, in3):
    return one_in_one_out


def code_two_func():
    return one_in_one_out, one_in_one_out

def code_two_func_in1(in1):
    return one_in_one_out, one_in_one_out

def code_two_func_in2(in1, in2):
    return one_in_one_out, one_in_one_out

def code_two_func_in3(in1, in2, in3):
    return one_in_one_out, one_in_one_out


def code_three_func(in1):
    return one_in_one_out, one_in_one_out, one_in_one_out


def code_no_func(in1):
    return 42

def train_no_func(in1, in2):
    return in1 + in2


def test_code_proc():
    # === VALID CASES ===
    # one in - one out
    out = coder(code_one_func_in1, "in1", "func").run({"in1": 1})
    assert out["func"] is one_in_one_out
    # two in - one out
    out = coder(code_one_func_in2, ["in1", "in2"], "func").run({"in1": 1, "in2": 2})
    assert out["func"] is one_in_one_out
    # one in - two out
    out = coder(code_two_func_in1, "in1", ["func1", "func2"]).run({"in1": 1})
    assert out["func1"] is one_in_one_out
    assert out["func2"] is one_in_one_out
    # two in - two out
    out = coder(code_two_func_in2, ["in1", "in2"], ["func1", "func2"]).run({"in1": 1, "in2": 2})
    assert out["func1"] is one_in_one_out
    assert out["func2"] is one_in_one_out
    # define with dict
    out = coder(code_one_func_in2, {"in1": "n1", "in2": "n2"}, "func").run({"n1": 1, "n2": 2})
    assert out["func"] is one_in_one_out

    # copy coder
    c1 = coder(code_two_func_in2, ["in1", "in2"], ["func1", "func2"])
    c2 = c1._copy()
    assert c1 == c2

    # === INVALID CASES ===
    # invalid in
    pattern = r"Invalid input type"
    with pytest.raises(ValueError, match=pattern):
        out = coder(code_one_func, None, "func").run()

    # too many outs
    pattern = r"Expected 1 or 2 outputs, found 3"
    with pytest.raises(ValueError, match=pattern):
        out = coder(code_three_func, "in1", ["func1", "func2", "func3"]).run()

    # no outs
    pattern = r"Invalid output type"
    with pytest.raises(ValueError, match=pattern):
        out = coder(code_one_func_in1, "in1", None).run()

    # invalid coder returning smt else
    pattern = r"Coder expected callable outputs but got <class 'int'> instead"
    with pytest.raises(ValueError, match=pattern):
        out = coder(code_no_func, "in1", "funcOut").run({"in1": 1})

    # incorrect run call
    pattern = r"Coder.run\(\) expects a dictionary, but got <class 'list'> instead"
    with pytest.raises(ValueError, match=pattern):
        out = coder(code_no_func, "in1", "funcOut").run(["in1"])


def test_processor():
    # first input is the function: one in - one out
    the_input = {"my_func": one_in_one_out, "dsIn": 42}
    out = processor(["my_func", "dsIn"], "dsOut").run(the_input)
    assert out["dsOut"] == 42

    # func with multiple inputs
    the_input = {"my_func": two_in_first_out, "dsIn1": 42, "dsIn2": 43}
    out = processor(["my_func", "dsIn1", "dsIn2"], "dsOut").run(the_input)
    assert out["dsOut"] == 42

    # single input - becomes no input
    out = processor("my_func", "dsOut").run({"my_func": zero_in_one_out})
    assert out["dsOut"] == 42

    # can chain coder & processor
    out1 = coder(code_one_func_in1, "in1", "func").run({"in1": 1})
    assert out1["func"] is one_in_one_out
    out1.update({"dsIn": 42})
    out2 = processor(["func", "dsIn"], "dsOut").run(out1)
    assert out2["dsOut"] == 42

    # copy processor
    p1 = processor(["my_func", "dsIn"], "dsOut")
    p2 = p1._copy()
    assert p1 == p2

    # ensure the input is not modified
    orig_input = {"my_func": one_in_one_out, "dsIn": 42}
    the_input = orig_input.copy()
    out = processor(["my_func", "dsIn"], "dsOut").run(the_input)
    assert out["dsOut"] == 42
    assert the_input == orig_input


    # === INVALID CASES ===
    # no input
    pattern = r"Invalid Processor definition: it must have some 'inputs'."
    with pytest.raises(ValueError, match=pattern):
        out = processor(None, None)

    # invalid input type
    pattern = r"Invalid Processor definition: 'inputs' type must be one of \[String, List\], not 'bool'."
    with pytest.raises(ValueError, match=pattern):
        out = processor(True, "dsOut")

    # output of wrong type
    pattern = r"Invalid Processor definition: 'outputs' type must be one of \[String, List, Dict, None\], not 'int'."
    with pytest.raises(ValueError, match=pattern):
        out = processor(["my_func", "dsIn"], 42)

    # empty input
    pattern = r"Invalid input size"
    with pytest.raises(AssertionError, match=pattern):
        out = processor(["my_func", "dsIn"], "dsOut")
        out.run({})
    
    # mismatch in input size
    pattern = r"Inputs of 'one_in_one_out' function expected \['arg'\], but got \['dsIn', 'dsIn1'\]"
    the_input = {"my_func": one_in_one_out, "dsIn": 42, "dsIn1": 29}
    with pytest.raises(TypeError, match=pattern):
        out = processor(["my_func", "dsIn"], "dsOut")
        out.run(the_input)

    # invalid processor name
    pattern = r"Invalid Processor definition: '42' is not a valid processor name."
    with pytest.raises(ValueError, match=pattern):
        out = processor(["my_func", "dsIn"], "dsOut", name = 42)

    # input is not function
    the_input = {"my_func": 40, "dsIn": 42}
    pattern = r"Invalid Processor definition: First 'input' must be a callable, not 'int'."
    with pytest.raises(ValueError, match=pattern):
        out = processor(["my_func", "dsIn"], "dsOut").run(the_input)

    # run without dict
    pattern = r"Processor.run\(\) expects a dictionary, but got \<class 'str'\> instead."
    with pytest.raises(ValueError, match=pattern):
        out = processor(["my_func", "dsIn"], "dsOut").run("foo")

    # processor with exception
    def raise_exception(arg):
        raise ValueError("Exception raised")
    the_input = {"my_func": raise_exception, "dsIn": 42}
    pattern = r"Exception raised"
    with pytest.raises(ValueError, match=pattern):
        out = processor(["my_func", "dsIn"], "dsOut").run(the_input)


def test_train():
    # two in - one out
    out = trainer(code_one_func_in2, ["in1", "in2"], "func").run({"in1": 1, "in2": 2})
    assert out["func"] is one_in_one_out

    # three in - one out
    out = trainer(train_one_func_in3, ["in1", "in2", "in3"], "func", numX = 2).run(
        {"in1": 1, "in2": 2, "in3": 3}
    )
    assert out["func"] is one_in_one_out

    # two in - one out with dict
    out = trainer(code_one_func_in2, {"in1": "n1", "in2": "n2"}, "func").run(
        {"n1": 1, "n2": 2})
    assert out["func"] is one_in_one_out


    # copy trainer
    out1 = trainer(train_one_func_in3, ["in1", "in2", "in3"], "func", numX = 1)
    out2 = out1._copy()
    assert out1 == out2


    # === INVALID CASES ===
    # missing input 
    pattern = r"At least two inputs required, found 1"
    with pytest.raises(ValueError, match=pattern):
        out = trainer(code_one_func_in1, ["in1"], "func")
    
    # input must be list
    pattern = r"Invalid input type"
    with pytest.raises(ValueError, match=pattern):
        out = trainer(code_one_func_in1, "in1", "func")

    # too many outputs - no out
    pattern = r"'outputs' type must be one of \[String, List\], not 'NoneType'."
    with pytest.raises(ValueError, match=pattern):
        out = trainer(code_one_func_in2, ["in1", "in2"], None)
        
    # callable output expected
    pattern = r"Trainer expected callable output but got <class 'int'> instead!"
    with pytest.raises(ValueError, match=pattern):
        out = trainer(train_no_func, ["in1", "in2"], "func").run({"in1": 1, "in2": 2})

    # incorrect run call
    pattern = r"Trainer.run\(\) expects a dictionary, but got <class 'list'> instead"
    with pytest.raises(ValueError, match=pattern):
        out = trainer(train_one_func_in3, ["in1", "in2", "in3"], "func", numX = 2).run(["in1"])

    # invalid trainer: two many X values 
    pattern = r"numX=2 must be <= 1 \(at least one Y is required\)"
    with pytest.raises(ValueError, match = pattern):
        trainer(code_one_func_in2, ["d1", "f2"], "func", numX = 2)


def test_code_check():
    # COMMON SETUP
    types : TypeCatalog = TypeCatalog()
    types.add_data("d1")
    types.add_data("d2")

    # valid coder - one out (assume coding...)
    c = coder(code_one_func_in1, "d1", "f1").check(types)
    assert types["f1"].is_function()
    assert types["f1"].in_type[0] == types["d1"]
    assert types["f1"].out_type[0].is_data()

    # valid coder w/ parameter
    c = coder(code_one_func_in2, ["d1", "parameters"], "f2").check(types)
    assert types["f2"].is_function()
    assert types["f2"].in_type[0] == types["d1"]
    assert "parameters" not in types
    assert types["f2"].out_type[0].is_data()

    # valid coder - two out
    c = coder(code_two_func_in1, "d1", ["f3", "f4"]).check(types)
    assert types["f3"].is_function()
    # f1: d1 -> d2 (encoding)
    assert types["f3"].in_type[0] == types["d1"]
    assert types["f3"].out_type[0].is_data()
    d2 = types["f3"].out_type[0]
    # f2: d2 -> d1 (decoding)
    assert types["f4"].is_function()
    assert types["f4"].in_type[0] == d2
    assert types["f4"].out_type[0] == types["d1"]

    # valid coder - two in
    c = coder(code_one_func_in2, ["d1", "d2"], "f5").check(types)
    assert types["f5"].is_function()
    assert types["f5"].in_type[0] == types["d1"]
    assert types["f5"].in_type[1] == types["d2"]
    assert types["f5"].out_type[0].is_data()

    # invalid coder - function in
    pattern = r"input f1 is not data"
    with pytest.warns(Warning, match=pattern):
        c = coder(code_one_func_in1, "f1", "d1")
        c.check(types)


def test_train_check():
    # COMMON SETUP
    types : TypeCatalog = TypeCatalog()
    types.add_data("d1")
    types.add_data("d2")
    types.add_data("d3")

    # valid trainer, func: (d1) -> (d2)
    trainer(code_one_func_in2, ["d1", "d2"], "f1").check(types)
    assert types["f1"].is_function()
    assert types["f1"].in_type[0] == types["d1"]
    assert types["f1"].out_type[0].is_data()
    assert types["f1"].out_type[0] == types["d2"]

    # valid trainer, func: (d1) -> (d2) w/ parameter
    trainer(train_one_func_in3, ["d1", "d2", "parameters"], "f2").check(types)
    assert types["f2"].is_function()
    assert types["f2"].in_type[0] == types["d1"]
    assert "parameters" not in types
    assert types["f2"].out_type[0].is_data()
    assert types["f2"].out_type[0] == types["d2"]

    # valid trainer, func: (d1, d2) -> (d3)
    out = trainer(train_one_func_in3, ["d1", "d2", "d3"], "f3", numX = 2).check(types)
    assert types["f3"].is_function()
    assert types["f3"].in_type[0] == types["d1"]
    assert types["f3"].in_type[1] == types["d2"]
    assert types["f3"].out_type[0] == types["d3"]

    # valid trainer, func: (d1) -> (d2, d3)
    out = trainer(train_one_func_in3, ["d1", "d2", "d3"], "f4", numX = 1).check(types)
    assert types["f4"].is_function()
    assert types["f4"].in_type[0] == types["d1"]
    assert types["f4"].out_type[0] == types["d2"]
    assert types["f4"].out_type[1] == types["d3"]

    # invalid trainer: first input is not data
    pattern = r"Data expected as X input 'f1'"
    with pytest.warns(Warning, match=pattern):
        trainer(code_one_func_in2, ["f1", "d2"], "f5").check(types)

    # invalid trainer: second input is not data
    pattern = r"Data expected as Y input 'f2'"
    with pytest.warns(Warning, match=pattern):
        trainer(code_one_func_in2, ["d1", "f2"], "f6").check(types)


def test_type_print():
    # COMMON SETUP
    types : TypeCatalog = TypeCatalog()
    types.add_data("d1")
    types.add_data("d2")
    types.add_function("f1", types["d1"], types["d2"])

    assert str(types["d1"]) == "DataType(1)"
    assert str(types["d2"]) == "DataType(2)"
    assert str(types["f1"]) == "FunctionType([DataType(1)], [DataType(2)])"

def test_process_check():
    # COMMON SETUP
    types : TypeCatalog = TypeCatalog()
    types.add_data("d1")
    types.add_data("d2")
    types.add_data("d3")
    types.add_function("f0", [], types["d2"])
    types.add_function("f1", types["d1"], types["d2"])
    types.add_function("f2", types["d2"], types["d1"])
    types.add_function("f3", [types["d1"], types["d2"]], types["d3"])
    types.add_function("f4", types["d1"], [types["d2"], types["d3"]])
    types.add_function("one_in_one_out", [], types["d1"])

    # zero in - one out
    processor("f0", "d0").check(types)

    # fixed_func w/ params 
    #p = processor("parameters", "d0", func = one_in_one_out)
    #p.check(types)

    # one in - one out
    out = processor(["f1", "d1"], "d9").check(types)
    assert types["d9"].is_data()
    assert types["d9"] == types["d2"]

    # two in - one out
    out = processor(["f3", "d1", "d2"], "d10").check(types)
    assert types["d10"].is_data()
    assert types["d10"] == types["d3"]

    # one in - two out
    out = processor(["f4", "d1"], ["d11", "d12"]).check(types)
    assert types["d11"].is_data()
    assert types["d11"] == types["d2"]
    assert types["d12"].is_data()
    assert types["d12"] == types["d3"]

    # invalid processor: mismatch in number of input
    pattern = r"inconsistent number of inputs, expected 2, got 3"
    with pytest.raises(ValueError, match=pattern):
        out = processor(["f3", "d1", "d2", "d3"], "d10").check(types)
    
    # invalid processor: mismatch in number of input
    pattern = r"inconsistent number of inputs, expected 2, got 1"
    with pytest.raises(ValueError, match=pattern):
        out = processor(["f3", "d1"], "d10").check(types)

    # invalid processor: mismatch in number of input
    pattern = r"inconsistent number of inputs, expected 1, got 2"
    with pytest.raises(ValueError, match=pattern):
        out = processor(["f1", "d1", "d2"], "d3").check(types)
    
    # invalid processor: mismatch in number of output
    pattern = r"inconsistent number of outputs, expected 1, got 2"
    with pytest.raises(ValueError, match=pattern):
        out = processor(["f1", "d1"], ["d2", "d3"]).check(types)
    
    # invalid processor: mismatch in number of output
    pattern = r"inconsistent number of outputs, expected 2, got 1"
    with pytest.raises(ValueError, match=pattern):
        out = processor(["f4", "d1"], ["d15"]).check(types)

    # invalid processor: first input is not function
    pattern = r"function expected as first input"
    with pytest.warns(Warning, match=pattern):
        out = processor(["d1", "d2"], "d3").check(types)

    # invalid processor: remaning inputs are not data
    pattern = r"data expected for input 'f2'"
    with pytest.warns(Warning, match=pattern):
        out = processor(["f1", "f2"], "d1").check(types)

    # invalid processor: function and data not compatible
    pattern = r"function 'f1' and data 'd2' are not compatible" 
    with pytest.warns(Warning, match=pattern):
        out = processor(["f1", "d2"], "d1")
        out.check(types)

    # invalid processor: output is not data
    pattern = r"inconsistent output type"
    with pytest.warns(Warning, match=pattern):
        out = processor(["f1", "d1"], "f2").check(types)

def test_processor_fixed():
    # process w/ fixed function 
    types : TypeCatalog = TypeCatalog()
    types.add_data("X")
    types.add_data("Y")

    from kedro_umbrella.library import split_data
    X = [1, 2, 3, 4, 5]
    Y = [2, 4, 6, 8, 10]

    # run the function
    the_input = {"X": X, "Y": Y, "parameters": {"random_state": 42}}
    #out = processor("X", "Y", func = split_data).check()
    split = processor(
                    inputs = ["X", "Y", "parameters"],
                    outputs = ["X_train", "X_test", "Y_train", "Y_test"],
                    func = split_data)

    # check the types
    split.check(types)
    assert types["X_train"].is_data()
    assert types["X_train"] == types["X"]
    assert types["X_test"].is_data()
    assert types["X_test"] == types["X"]

    assert types["Y_train"].is_data()
    assert types["Y_train"] == types["Y"]
    assert types["Y_test"].is_data()
    assert types["Y_test"] == types["Y"]

    out = split.run(the_input)
    assert len(out["X_train"]) == 3
    assert len(out["X_test"]) == 2
    assert len(out["Y_train"]) == 3
    assert len(out["Y_test"]) == 2

    # invalid fixed function
    pattern = r"'func' must be a callable, not 'str'"
    with pytest.raises(ValueError, match=pattern):
        split = processor(
                    inputs = ["X", "Y"],
                    outputs = ["X_train", "X_test", "Y_train", "Y_test"],
                    func = "split_data")




