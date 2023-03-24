# Better Practices for Domain Adaptation
## UDA

### Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

### Set environment variables
Set the following variables in your terminal:
```bash
export UDA_ROOT=<path-to-this-directory>

export DATA_ROOT=<path-to-data-directory>
export RESULTS_ROOT=$UDA_ROOT/results
```

### Datasets
To download datasets:
Download VisDA2017 manually from [here](https://ai.bu.edu/visda-2017/).
Download Office-31 manually from [here](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code).
Download Office-Home manually from [here](https://www.hemanthdv.org/officeHomeDataset.html).

To generate MNIST-MR, run:
```bash
mkdir ${DATA_ROOT}/bsds500 && cd ${DATA_ROOT}/bsds500 && wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz && cd ${UDA_ROOT}
python generate_mnistmr.py
```

Next we generate the data split files for VisDA2017/Office-31/OfficeHome by running:
```bash
python split.py ${DATA_ROOT}/VisDA2017/classification train 0.6 0.2
python split.py ${DATA_ROOT}/VisDA2017/classification validation 0.6 0.2
python split.py ${DATA_ROOT}/VisDA2017/classification test 0.6 0.2

python split.py ${DATA_ROOT}/office31 amazon 0.6 0.2
python split.py ${DATA_ROOT}/office31 dslr 0.6 0.2
python split.py ${DATA_ROOT}/office31 webcam 0.6 0.2

python split.py ${DATA_ROOT}/officehome art 0.6 0.2
python split.py ${DATA_ROOT}/officehome clipart 0.6 0.2
python split.py ${DATA_ROOT}/officehome product 0.6 0.2
python split.py ${DATA_ROOT}/officehome real 0.6 0.2
```

When all is done, the datasets are expected in the following filestructure:
```bash
${DATA_ROOT}/
    mnist_m_r/
        ...
    VisDA2017/
        ...
    office31/
        ...
    officehome/
        ...
```

### Source-only Models
For the training, we will run through a setup on MNIST-M. The other datasets can be used by specifying the right `--dataset`, `--source` and `--target` arguments.

To train 10 source-only models with different hyperparameters and select the checkpoint to use as initialisation for adaptation, run the following:
```bash
python classification_benchmark.py --data-root ${DATA_ROOT} --results-root ${RESULTS-ROOT} --dataset mnistm --source mnist --target mnistm --algorithm source-only --G-arch mnistG --hpo-num-samples 10 --hpo-validate-freq 5 --hpo-max-epochs 100
python select_best_checkpoint ${RESULTS_ROOT} mnistm mnist mnistm source-only src_val_acc_score
```

The checkpoint should then exist in the following structure:
```bash
${RESULTS_ROOT}/
    mnistm/
        mnist/
            mnistm/
                source-only/
                    best.pt
                    ...
```

### Adaptation
To run all the adaptation algorithms, using the generated source-only checkpoint as initialisation, run the following:
```bash
python classification_benchmark.py --data-root ${DATA_ROOT} --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm atdoc --init-source-only --G-arch mnistG --hpo-num-samples 10 --hpo-validate-freq 5 --hpo-max-epochs 100
python classification_benchmark.py --data-root ${DATA_ROOT} --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm bnm --init-source-only --G-arch mnistG --hpo-num-samples 10 --hpo-validate-freq 5 --hpo-max-epochs 100
python classification_benchmark.py --data-root ${DATA_ROOT} --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm dann --init-source-only --G-arch mnistG --hpo-num-samples 10 --hpo-validate-freq 5 --hpo-max-epochs 100
python classification_benchmark.py --data-root ${DATA_ROOT} --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm mcc --init-source-only --G-arch mnistG --hpo-num-samples 10 --hpo-validate-freq 5 --hpo-max-epochs 100
python classification_benchmark.py --data-root ${DATA_ROOT} --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm mcd --init-source-only --G-arch mnistG --hpo-num-samples 10 --hpo-validate-freq 5 --hpo-max-epochs 100
python classification_benchmark.py --data-root ${DATA_ROOT} --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm mmd --init-source-only --G-arch mnistG --hpo-num-samples 10 --hpo-validate-freq 5 --hpo-max-epochs 100
```

### Compute Validators
Next, we compute all validators for the algorithm checkpoints by running the following:
```bash
python compute_validators.py --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm source-only
python compute_validators.py --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm atdoc
python compute_validators.py --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm bnm
python compute_validators.py --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm dann
python compute_validators.py --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm mcc
python compute_validators.py --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm mcd
python compute_validators.py --results-root ${RESULTS_ROOT} --dataset mnistm --source mnist --target mnistm --algorithm mmd
```

### Generate tables and figures
Open the jupyter notebook `generate_table.py` and run all cells. A latex table similar to table 3 will be generated, containing the results over the datasets used.


### Acknowledgements

Our code is built upon the public code of the [pytorch-adapt](https://github.com/vita-epfl/ttt-plus-plus).
