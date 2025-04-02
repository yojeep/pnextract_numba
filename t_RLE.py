import numba as nb
from numba import int64, types

# 1. First, create a deferred type for the self-referential field
MyStruct_type = types.DeferredType()

# 2. Define the spec with the deferred type
spec = [
    ('x', int64),
    ('y', int64),
    ('z', int64),
    ('nei', types.ListType(int64)),
    ('boss', types.Optional(MyStruct_type))  # Can be None or another MyStruct
]

# 3. Define the class
@nb.experimental.jitclass(spec)
class MyStruct:
    def __init__(self, x, y, z, nei, boss=None):
        self.x = x
        self.y = y
        self.z = z
        self.nei = nei
        self.boss = boss

# 4. Now tell Numba that the deferred type is actually MyStruct
MyStruct_type.define(MyStruct.class_type.instance_type)
# 5. Create an empty typed list for initialization
empty_list = nb.typed.List.empty_list(int64)
MyStructType = MyStruct.class_type.instance_type
ball_space = nb.typed.List.empty_list(MyStructType)

# Example usage
boss = MyStruct(1, 2, 3, empty_list)
employee = MyStruct(4, 5, 6, empty_list, boss)
for i in range(1000*1000*1000):
    ball_space.append(MyStruct(i, i+1, i+2, empty_list, boss))
boss.x = 1000
print(employee.boss.x)