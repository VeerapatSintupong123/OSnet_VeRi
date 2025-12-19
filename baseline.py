import torchreid
from dataset import VeRiDataset

if __name__ == '__main__':

    print(f"torchreid versions is: {torchreid.__version__}")

    try:
        torchreid.data.register_image_dataset('synthetic_veri_reid', VeRiDataset)
    except Exception as e:
        print(e)

    datamanager = torchreid.data.ImageDataManager(
        sources='synthetic_veri_reid',
        height = 256,
        width = 256,
        transforms =['random_flip', 'random_crop']
    )

    model = torchreid.models.build_model(
        name='osnet_x0_25',
        num_classes=datamanager.num_train_pids,
        loss='triplet'
    )

    model = model.cuda()

    print(next(model.parameters()).is_cuda)

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    engine.run(
        save_dir="log/osnet_x0_25",
        normalize_feature=True,
        test_only=True,
        visrank=True,
        visrank_topk=5
    )

# ** Results osnet_x0_25**
# mAP: 83.6% 
# CMC curve 
# Rank-1 : 92.4%
# Rank-5 : 97.4%
# Rank-10 : 97.9%
# Rank-20 : 98.8%

# ** Results osnet_x0_5**
# mAP: 83.4%
# CMC curve
# Rank-1  : 92.7%
# CMC curve
# Rank-1  : 92.7%
# Rank-5  : 97.5%
# Rank-10 : 98.0%
# Rank-20 : 98.6%