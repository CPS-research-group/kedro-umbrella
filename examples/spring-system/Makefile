all: run check

# run full train/test pipeline
run:
	kedro run
	diff -r ./data/08_reporting_gold ./data/08_reporting

# check pipeline
check:
	kedro check
