all: \
	 pipeline_full  \
	 pipeline_sensitivity \
	 pipeline_type_ko \
	 pipeline_type_ok

define clean
	rm -rf src/pipe_deformation/__pycache__
	rm -rf ./data/08_reporting/*
endef

export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

pipeline_full:
	$(call clean)
	cd src/pipe_deformation && ln -sf pipeline_full.py pipeline.py
	kedro run
	diff -r ./data/08_reporting_gold ./data/08_reporting

pipeline_sensitivity:
	$(call clean)
	cd src/pipe_deformation && ln -sf pipeline_sensitivity.py pipeline.py
	kedro run

pipeline_compose:
	$(call clean)
	cd src/pipe_deformation && ln -sf pipeline_compose.py pipeline.py
	kedro run

pipeline_type_ko:
	$(call clean)
	cd src/pipe_deformation && ln -sf pipeline_type_error.py pipeline.py
	kedro check

pipeline_type_ok:
	$(call clean)
	cd src/pipe_deformation && ln -sf pipeline_full.py pipeline.py
	kedro check
