# self.seeds.clear()
        # self.lineids.clear()
        # self.linepts.clear()
        # qualities = vtk.vtkFloatArray()
        # qualities.SetName("Quality")
        # arr:vtk.vtkVariantArray = table.GetColumnByName("ImageData")
        # idx = 0
        # concat_pts = vtk.vtkPoints()
        # for i in range(arr.GetNumberOfValues()):
        #     frame_seeds = vtk.vtkPoints()
        #     imageData = arr.GetValue(i).ToVTKObject()
        #     if not isinstance(imageData, vtk.vtkImageData):
        #         print(f"Wrong type (ImageData needed):", type(imageData))
        #         return 1
        #     frame_pts, frame_lineids_list = self.generate_streamlines(imageData, idx)
        #     concat_pts.InsertPoints(concat_pts.GetNumberOfPoints(), frame_pts.GetNumberOfPoints(), 0, frame_pts)
        #     for frame_lineids in frame_lineids_list:
        #         output.InsertNextCell(vtk.VTK_POLY_LINE, frame_lineids)
        #         firstid = frame_lineids.GetId(0) - idx
        #         firstpoint = frame_pts.GetPoint(firstid)
        #         frame_seeds.InsertNextPoint(firstpoint) # we take the 1st point (furthest in backward direction) as seed
        #     self.seeds[i] = frame_seeds
        #     self.lineids[i] = frame_lineids_list
        #     self.linepts[i] = frame_pts
        #     qualities.InsertNextValue(self.measure_quality(i))
        #     idx += frame_pts.GetNumberOfPoints()
        
        # output.SetPoints(concat_pts)
        # fielddata:vtk.vtkFieldData = output.GetFieldData()
        # fielddata.AddArray(qualities)


# vector field source point -> vector conversion
# arr = np.flip(self.np_pts, 1) # reorder the array due to meshgrid order differing from imagedata
# arr_slice = arr[0:(self.xsize*self.ysize)] # truncate to x0,...,xn,y0,...,yn
# arr = np.tile(arr_slice, (self.zsize, 1)) # tile in z plane, self.zsize number of times
# return self.rotate_z(arr, self.z_angle)