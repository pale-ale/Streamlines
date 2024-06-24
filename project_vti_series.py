import vtk
import glob
import os

def process_vti_series(input_pattern, output_dir, slice_plane='xy', slice_index=0, base_name="frame"):
    # find vti files in given directory
    files = sorted(glob.glob(input_pattern))
    
    # determine the required zero-padding length based on the number of files
    num_files = len(files)
    padding_length = len(str(num_files))
    
    # check output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, input_file in enumerate(files):
        # read vti files
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(input_file)
        reader.Update()
        
        # get the data from the reader
        image_data = reader.GetOutput()
        
        # get the number of components (assuming 3)
        num_components = image_data.GetPointData().GetArray("data").GetNumberOfComponents()
        if num_components != 3:
            raise ValueError(f"Expected 3 components, but got {num_components}")
        
        # get the scalar data (here named "data")
        scalar_data = image_data.GetPointData().GetArray("data")
        
        # create projected data
        new_scalar_data = vtk.vtkFloatArray()
        new_scalar_data.SetName("data")
        new_scalar_data.SetNumberOfComponents(3)
        new_scalar_data.SetNumberOfTuples(scalar_data.GetNumberOfTuples())
        
        # iterate through each point and set the third component to 0
        for i in range(scalar_data.GetNumberOfTuples()):
            tuple_value = list(scalar_data.GetTuple3(i))
            tuple_value[2] = 0
            new_scalar_data.SetTuple(i, tuple_value)
        
        # set the new scalar data back to the image
        image_data.GetPointData().AddArray(new_scalar_data)
        
        # extract the slice
        slicer = vtk.vtkExtractVOI()
        slicer.SetInputData(image_data)
        
        dimensions = image_data.GetDimensions()
        
        if slice_plane == 'xy':
            slicer.SetVOI(0, dimensions[0]-1, 0, dimensions[1]-1, slice_index, slice_index)
        elif slice_plane == 'xz':
            slicer.SetVOI(0, dimensions[0]-1, slice_index, slice_index, 0, dimensions[2]-1)
        elif slice_plane == 'yz':
            slicer.SetVOI(slice_index, slice_index, 0, dimensions[1]-1, 0, dimensions[2]-1)
        else:
            raise ValueError(f"Unknown slice plane: {slice_plane}")
        
        slicer.Update()
        sliced_image_data = slicer.GetOutput()
        
        # write the modified data to a new file
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(sliced_image_data)
        writer.Write()

input_pattern = "./HotRoom3D-2/*.vti"
output_dir = "./HotRoom2D/"
slice_plane = 'xy'  # can be 'xy', 'xz', or 'yz'
slice_index = 30     # index of the slice along the selected plane
base_name = "frame" # base name for output files

process_vti_series(input_pattern, output_dir, slice_plane, slice_index, base_name)
