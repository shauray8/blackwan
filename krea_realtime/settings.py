import dotenv
import os
dotenv.load_dotenv()

MODEL_FOLDER = os.getenv("MODEL_FOLDER", "wan_models")
COMPILE_SHAPES = [(832, 480), (480, 832)]