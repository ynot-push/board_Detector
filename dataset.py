import pandas as pd
import PIL
from PIL  import Image,ImageDraw
from pandas import read_csv
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pathlib
import cv2

class BoardDataset(Dataset):
    def __init__(self,csv_path,image_dir):
        self.df=pd.read_csv(csv_path)
        self.image_dir=pathlib.Path(image_dir)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    def __len__(self):
            return len(self.df)    
    
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        path=self.image_dir / row["filename"]
        img=Image.open(path).convert("RGB")
        w,h=img.size
        x1,y1,x2,y2=row["x1"],row["y1"],row["x2"],row["y2"]
        box=[x1/w, y1/h, x2/w, y2/h]
        box=torch.tensor(box,dtype=torch.float32)
        has_board = torch.tensor([float(row["has_board"])], dtype=torch.float32)
        img=self.transform(img)

        return img,box,has_board



if(__name__ =="__main__") :
    ds = BoardDataset("dataset/labels.csv", "dataset/images")
    print(len(ds))
    print(ds.df.iloc[0])
    path=ds.image_dir / ds.df.iloc[0]["filename"]
    img=Image.open(path)
    print(img.size)

    print("###############################################################")

   
    img2, box2, has_board2 = ds[27]
    print(img2.shape, has_board2, box2)

    image = transforms.ToPILImage()(img2)

    # box1 is [x1/w, y1/h, x2/w, y2/h]
    x1 = box2[0].item() * 256
    y1 = box2[1].item() * 256
    x2 = box2[2].item() * 256
    y2 = box2[3].item() * 256

    # Draw bounding box
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3
    )
    image.save("debug_box.png")

    print("Saved debug image to debug_box.png")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    imgs, boxes, flags = next(iter(loader))
    print(imgs.shape, boxes.shape, flags.shape)