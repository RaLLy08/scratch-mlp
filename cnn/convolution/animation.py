import tkinter as tk
from PIL import Image, ImageTk

# Function to update zoomed-in image in the zoom box
def update_zoom(event):
    # Get the current mouse position
    x, y = event.x, event.y
    
    # Ensure the cursor is within the bounds of the image
    if 0 <= x <= img_width and 0 <= y <= img_height:
        # Define the size of the zoomed-in region
        zoom_box_size = 100
        zoom_level = 4
        
        # Calculate the bounds for the zoomed-in region
        left = max(x - zoom_box_size // 2, 0)
        top = max(y - zoom_box_size // 2, 0)
        right = min(left + zoom_box_size, img_width)
        bottom = min(top + zoom_box_size, img_height)
        
        # Crop the zoomed-in region from the original image
        zoomed_region = img.crop((left, top, right, bottom))
        
        # Resize the zoomed region to simulate zooming
        zoomed_region = zoomed_region.resize(
            (zoom_box_size * zoom_level, zoom_box_size * zoom_level), Image.Resampling.LANCZOS
        )
        
        # Convert the zoomed region to an ImageTk format for displaying
        zoomed_tk = ImageTk.PhotoImage(zoomed_region)
        
        # Update the label displaying the zoomed-in region
        zoom_label.config(image=zoomed_tk)
        zoom_label.image = zoomed_tk

# Initialize the Tkinter window
root = tk.Tk()
root.title("Image Zoom Tool")


# Load the image
img_path = "./1581884.png"  # Replace with your image path
img = Image.open(img_path)
img_width, img_height = img.size
img_tk = ImageTk.PhotoImage(img)

# Create a canvas to display the image
canvas = tk.Canvas(root, width=img_width, height=img_height)
canvas.pack()

# Add the image to the canvas
canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

# Create a label to display the zoomed-in portion
zoom_label = tk.Label(root, width=120, height=120)
zoom_label.place(x=0, y=0)

# Bind mouse motion to update the zoomed-in region
canvas.bind("<Motion>", update_zoom)

# Run the Tkinter main loop
root.mainloop()
