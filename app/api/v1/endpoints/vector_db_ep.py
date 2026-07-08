from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO

router = APIRouter()

def get_vector_db(request: Request):
    return request.app.state.vector_db

@router.post('/vector_db')
def retrieve_from_edited_image(
    image: UploadFile = File(...),
    top_k: int = 5,
    vector_db=Depends(get_vector_db),
):
    try:
        # Read uploaded file and convert to PIL Image
        image_bytes = image.file.read()
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Search in vector database
        scores = vector_db.search_by_image(pil_image, top_k=top_k)
        
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    