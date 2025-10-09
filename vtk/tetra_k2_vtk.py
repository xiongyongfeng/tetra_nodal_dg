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
cells_connectivity_original = [0, 1, 2, 3, 4, 1, 0, 3]

# 创建点集和标量数组
all_points = vtk.vtkPoints()
density_array = vtk.vtkDoubleArray()
density_array.SetName("Density")
density_array.SetNumberOfComponents(1)

# 用于存储每个单元的点ID（线性顶点+边中点）
cell_point_ids_list = []

# 为每个单元创建点
for icell in range(2):
    point_ids_for_cell = []

    # 添加4个角点
    for i in range(4):
        vertex_idx = cells_connectivity_original[icell * 4 + i]
        point_id = all_points.InsertNextPoint(original_vertices[vertex_idx])
        point_ids_for_cell.append(point_id)
        density_array.InsertNextValue(1.0)  # 角点密度设为1.0

    # 计算并添加6个边中点
    edge_midpoints = []
    # 定义四面体的6条边（连接角点的索引对）
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]

    for edge in edges:
        pt1 = np.array(
            original_vertices[cells_connectivity_original[icell * 4 + edge[0]]]
        )
        pt2 = np.array(
            original_vertices[cells_connectivity_original[icell * 4 + edge[1]]]
        )
        midpoint = (pt1 + pt2) / 2.0
        point_id = all_points.InsertNextPoint(midpoint)
        point_ids_for_cell.append(point_id)
        density_array.InsertNextValue(0.5)  # 边中点密度设为0.5
        edge_midpoints.append(midpoint)

    cell_point_ids_list.append(point_ids_for_cell)

# 创建UnstructuredGrid并设置点集
ugrid = vtk.vtkUnstructuredGrid()
ugrid.SetPoints(all_points)
ugrid.GetPointData().SetScalars(density_array)

# 创建二阶四面体单元
for icell in range(2):
    quadratic_tetra = vtk.vtkQuadraticTetra()

    # 设置10个节点：4个角点 + 6个边中点
    for i in range(10):  # 二阶四面体有10个节点
        quadratic_tetra.GetPointIds().SetId(i, cell_point_ids_list[icell][i])

    ugrid.InsertNextCell(quadratic_tetra.GetCellType(), quadratic_tetra.GetPointIds())

# 写入VTU文件
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("TwoQuadraticTetra.vtu")
writer.SetInputData(ugrid)
writer.SetDataModeToAscii()
writer.Write()

print("VTU文件已生成: TwoQuadraticTetra.vtu")
print("包含两个二阶四面体单元，每个单元有10个节点。")
print("密度数据作为点数据存储（角点为1.0，边中点为0.5）。")
