all: run_noFMU check_noFMU

# run full train/test pipeline without FMU
run_noFMU:
	kedro run
	diff -r ./data/08_reporting_gold ./data/08_reporting

# check pipeline no FMU
check_noFMU:
	kedro check

# run training pipeline and export FMU
train_FMU:
	cd src/spring_system && ln -sf fmu_train_pipeline.py pipeline.py
	kedro run

# run test pipeline and import model from FMU
test_FMU:
	cd src/spring_system && ln -sf fmu_sim_pipeline.py pipeline.py
	kedro run
