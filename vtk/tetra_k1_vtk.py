import vtk
import numpy as np

# 原始网格顶点坐标
original_vertices = np.array(
    [
        [-1.0, -1.0, -1.0],  # 原始点0
        [1.0, -1.0, -1.0],  # 原始点1
        [-1.0, 1.0, -1.0],  # 原始点2
        [-1.0, -1.0, 1.0],  # 原始点3
        [-1.0, -3.0, -1.0],  # 原始点4
    ]
)

# 单元拓扑连接关系（引用原始顶点索引）
# 单元0: 顶点 [0, 1, 2, 3]
# 单元1: 顶点 [4, 1, 0, 3]
offset = 4
cells_connectivity_orignal = [0, 1, 2, 3, 4, 1, 0, 3]

# 为每个单元创建独立的点集（允许重复）
all_points = vtk.vtkPoints()

# 为每个单元的高阶点创建密度数据数组（作为点数据）
high_order_density_array = vtk.vtkDoubleArray()
high_order_density_array.SetName("HighOrderDensity")
high_order_density_array.SetNumberOfComponents(1)

# 存储每个单元的点在全局点列表中的新ID
cell_point_ids = []  # 用于存储单元0的4个顶点的新ID

for icel in range(2):
    # 为单元0添加顶点（对应原始顶点0,1,2,3）
    for i in range(4):
        point_id = all_points.InsertNextPoint(
            original_vertices[cells_connectivity_orignal[i + icel * offset]]
        )
        cell_point_ids.append(point_id)
        # 为每个几何顶点设置一个默认的高阶密度值（例如1.0）
        high_order_density_array.InsertNextValue(i)

# 创建UnstructuredGrid并设置点集
ugrid = vtk.vtkUnstructuredGrid()
ugrid.SetPoints(all_points)

for icel in range(2):
    # 创建单元0 (四面体)，使用单元0的新点ID
    tetra0 = vtk.vtkTetra()
    for i in range(4):
        tetra0.GetPointIds().SetId(i, cell_point_ids[i + icel * offset])
    ugrid.InsertNextCell(tetra0.GetCellType(), tetra0.GetPointIds())

# 将高阶点密度数据设置为点数据
ugrid.GetPointData().SetScalars(high_order_density_array)

# 写入VTU文件
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("TwoTetraK1WithDuplicatePoints.vtu")
writer.SetInputData(ugrid)
writer.SetDataModeToAscii()
writer.Write()

print("VTU文件已生成: TwoTetraK1WithDuplicatePoints.vtu")
print("点已重新编号并允许重复。")
print("高阶点密度数据作为点数据存储。")
