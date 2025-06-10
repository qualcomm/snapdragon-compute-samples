import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from model_loader import ModelLoader
from model_inference import ModelInference

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
    )

logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(description="HRNET Model App: Run main.py with or without arguments")

    parser.add_argument("--system", type=str, default="windows")
    parser.add_argument("--model", type=str, default="hrnet_pose")
    parser.add_argument("--processor", type=str, default="cpu")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--available_cameras", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="default")

    args = parser.parse_args()

    iLoad = ModelLoader(model=args.model, processor=args.processor, 
                        model_type=args.model_type)

    session = iLoad.load_model(iLoad.graphs)
    iInfer = ModelInference(session=session)

    if args.available_cameras:
        logger.info(iInfer.available_cameras)

    iInfer.inference(camera=args.camera)

if __name__=="__main__":
    main()
