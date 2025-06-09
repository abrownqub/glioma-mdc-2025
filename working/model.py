import config
import efficientnet_pytorch
import torch

# from efficientnet_pytorch import EfficientNet  # Efficientnet version b4


class HPV_Classifier(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

        self.features = efficientnet_pytorch.EfficientNet.from_pretrained(
            "efficientnet-b4",
            dropout_rate=config.DROPOUT,  # dropout rate before final classifier
            drop_connect_rate=config.DROPOUT,  # dropout rate at skip connections
        )

        # map features to binary decision
        # i've added some more layers to the classifier to see if it helps
        # this adds xxxxx parameters compared to 1792 parameters in the original !
        # thats a lot !!

        FEATURE_DIM = 1792
        STEP01_DIM = 2038
        STEP02_DIM = 1024
        STEP03_DIM = 512
        STEP04_DIM = 256
        OUTPUT_DIM = 1

        self.output_conv = torch.nn.Conv2d(FEATURE_DIM, STEP01_DIM, (7, 7))

        self.logits = torch.nn.Linear(STEP01_DIM, OUTPUT_DIM)

        # self.logits = torch.nn.Sequential(
        #     torch.nn.Linear(STEP01_DIM, STEP02_DIM),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(config.DROPOUT),
        #     torch.nn.Linear(STEP02_DIM, STEP03_DIM),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(config.DROPOUT),
        #     torch.nn.Linear(STEP03_DIM, STEP04_DIM),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(config.DROPOUT),
        #     torch.nn.Linear(STEP04_DIM, OUTPUT_DIM),
        # )

    def freeze_features(
        self,
    ):

        # freeze the features
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_features(
        self,
    ):

        # unfreeze the features
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(
        self,
        X,  # B, 3, 244, 244
    ):

        # get features from pretrained network
        X = self.features.extract_features(
            X,
        )  # B, 1792, 7, 7

        X = self.output_conv(X)  # B, 1792, 1, 1
        X = torch.squeeze(X)  # B, 1792
        X = torch.nn.functional.gelu(X)  # B, 1792
        X = torch.nn.functional.dropout(X, config.DROPOUT)  # B, 1792

        X = self.logits(X)  # B, 1

        return X  # B, 1

        # dynamically reduce the spatial dimensions of 'z' to be 1x1
        # z_pooled = torch.nn.functional.adaptive_avg_pool2d(X, 1)

    def save(
        self,
        path,
    ):
        torch.save(self.state_dict(), path)

    def load(
        self,
        path,
    ):
        print(f"loading model from {path}")
        # which device we are on
        device = self.features._conv_stem.weight.device
        try:
            self.load_state_dict(
                torch.load(
                    path,
                    map_location=device,
                ),
                strict=False,
            )
        except:
            print(f"failed to load model from {path}")

    def load_feature_extractor(
        self,
        path,
    ):
        # which device we are on
        device = self.features._conv_stem.weight.device
        try:
            self.features.load_state_dict(
                torch.load(
                    path,
                    map_location=device,
                ),
                strict=False,
            )
        except Exception as e:
            print(f"failed to load model from {path}: {e}")
