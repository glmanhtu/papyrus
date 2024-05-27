# Papyrus image similarity estimation

## Dataset preparation

Download the Geshaem dataset and save it to `/path/to/dataset`.
### Step 1: Segmentation
Assume the segmented dataset you want to save to is `/path/to/segmented`,
run the following script to segment:
```bash
git clone https://github.com/glmanhtu/PapyrusSegmentation.git
cd PapyrusSegmentation
pip install -r requirements.txt
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd ..
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth
python3 papyrus_segmentation.py --dataset_path /path/to/dataset --output_path /path/to/output
cd ..
```
### Step 2: Standardization
Assume the standardized dataset you want to save is `/path/to/standardized`, run the following 
script to standardize:
```bash
git clone https://github.com/glmanhtu/homer-competition.git
cd homer-competition
pip install -r requirements.txt
mkdir pretrained && cd pretrained
wget https://github.com/glmanhtu/homer-competition/releases/download/v1.0.0/first_twin-net.pth
cd ..
python standarize_images.py --pretrained_model_path pretrained --ref_box_height 24 --image_size 800 --dataset /path/to/segmented --prediction_path /path/to/standardized --cuda --p1_arch resnet50 --p2_arch resnet50
cd ..
```

### Step 3: Generate patches
```bash
python scripts/generate_michigan_patches.py --data-path /path/to/standardized --output-path /path/to/patches
```

## Training
Assume that the location of the preprocessed Geshaem dataset is `/path/to/patches`. Run the 
following script to train the deep learning model:
```bash
python main.py exp.tracking_uri=file:./output data.path=/path/to/patches run.name=triplet-vitS16-Dt2 data.version=2 data.batch_size=64 data.data_repeat=15 data.m_per_class=3 data=geshaem model=vit train.weight_decay=0.9 train.epochs=60 train.base_lr=1e-5 lr_scheduler.min_lr=1e-6 exp.name=Geshaem save_freq=10 tracking_enable=False
```

After training, pretrained model can be found at `output/models/model/best/checkpoint.pth`

## Generating similarity matrix
Once the model is fully trained, we can generate similarity matrix by running the following command:
```bash
python main.py exp.tracking_uri=file:./output data.path=/path/to/patches run.name=triplet-vitS16-Dt2 data.version=2 data.batch_size=64 data=geshaem model=vit exp.name=Geshaem model.weights=models/model/best mode=test
```

Pretrained model can also be downloaded from the release section,

## License
Licensed under the [GPLv3](https://github.com/glmanhtu/papyrus/blob/main/LICENSE) license.
