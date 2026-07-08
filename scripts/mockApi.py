import asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

MOCK_RESPONSE = {
    "approach_1": {
        "query": [
            "Tweed jacket in deep brown, paired with high-waisted trousers in matching brown, accented with dark brown belt and buckle, worn with sturdy black boots.",
            "Woolen sweater in forest green, layered over a cotton shirt in moss green, tucked into slim-fit khaki pants, completed with olive green canvas shoes.",
            "Denim jacket in navy blue, worn over a white t-shirt, paired with straight-leg jeans in dark indigo, finished with black leather boots with laces.",
            "Turtleneck sweater in burnt orange, layered with a wool vest in rust orange, worn with tailored trousers in burnt orange, and brown suede loafers.",
            "Plaid shirt in olive and mustard, worn with wide-leg trousers in olive, tucked into black ankle boots with metal buckles."
        ],
        "result": {
            "name_clothes": "m5_brown_7",
            "similarity": 0.26721322536468506,
            "best_description": "Woolen sweater in forest green, layered over a cotton shirt in moss green, tucked into slim-fit khaki pants, completed with olive green canvas shoes."
        }
    },
    "approach_2": {
        "result": {
            "name_clothes": "m6_brown_1",
            "similarity": 0.7129285335540771,
            "best_description": "Wool coat in deep forest green, worn over a charcoal gray sweater, paired with dark gray trousers, and black leather boots with rounded toes."
        }
    },
    "approach_3": {
        "background_caption": "This vibrant, sunlit Japanese village scene with wooden architecture, greenery, and a winding stone staircase suggests casual, comfortable attire suitable for a relaxed day of exploration in warm, pleasant weather, with earthy tones and traditional aesthetics guiding the choice.",
        "result": {
            "name_clothes": "m6_brown_2",
            "similarity": 0,
            "best_description": ""
        }
    }
}


@app.post("/mock-api")
async def mock_analyze(image: UploadFile = File(...)):
    # You can ignore the file completely
    await asyncio.sleep(10)
    return JSONResponse(content=MOCK_RESPONSE)
