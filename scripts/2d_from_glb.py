import os
from app.utils.mesh_2d import render_glb_front_view
from tqdm import tqdm

def main(
    input_folder,
    output_folder,
    image_size=(600, 1200),
    camera_distance_factor=1.0
):
    os.makedirs(output_folder, exist_ok=True)

    glb_files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".glb")
    )

    if not glb_files:
        print("No .glb files found.")
        return

    for glb_file in tqdm(glb_files, desc="Rendering GLB files", unit="file"):
        glb_path = os.path.join(input_folder, glb_file)

        png_name = os.path.splitext(glb_file)[0] + ".png"
        output_path = os.path.join(output_folder, png_name)

        try:
            render_glb_front_view(
                glb_path=glb_path,
                output_image_path=output_path,
                image_size=image_size,
                camera_distance_factor=camera_distance_factor
            )
        except Exception as e:
            print(f"Failed to render {glb_file}: {e}")

    print("✅ Rendering complete.")

if __name__ == "__main__":
    main(
        input_folder="data/glb",      # 👈 folder with .glb files
        output_folder="data/2d", # 👈 where .png files go
        image_size=(1200, 1200),
        camera_distance_factor=1.0
    )
