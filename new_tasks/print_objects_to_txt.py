from libero.libero.envs.objects import get_object_dict, get_object_fn

# Get a dictionary of all the objects
object_dict = get_object_dict()
# print(object_dict)
for key, value in object_dict.items():
    print(f"{key}: {value}")