import xmltodict

def getMorphologyStructure(xml_file):
    """
    Return the joint names and motor names in pre-order traversal of the given xml file.
    """
    return getOrderedJointNames(xml_file), getOrderedMotorNames(xml_file)

def getOrderedJointNames(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the joint names"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""

    def preorder(b):
        if "joint" in b:
            if isinstance(b["joint"], list) and b["@name"] != "torso":
                raise Exception(
                    "The given xml file does not follow the standard MuJoCo format."
                )
            elif not isinstance(b["joint"], list):
                b["joint"] = [b["joint"]]
            for j in b["joint"]:
                joint_names.append(j["@name"])
        if "body" not in b:
            return
        if not isinstance(b["body"], list):
            b["body"] = [b["body"]]
        for branch in b["body"]:
            preorder(branch)

    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joint_names = []
    try:
        root = xml["mujoco"]["worldbody"]["body"]
    except:
        raise Exception(
            "The given xml file does not follow the standard MuJoCo format."
        )
    preorder(root)
    return joint_names

def getOrderedMotorNames(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    motors = xml["mujoco"]["actuator"]["motor"]
    if not isinstance(motors, list):
        motors = [motors]
    for m in motors:
        joints.append(m["@joint"])
    return joints