import open3d as o3d
from rinarak.envs.gripper_env import GripperSimulator

if __name__ == "__main__":
    # Create simulator with integrated contact modeling
    sim = GripperSimulator(gui=False, auto_register=True)
    
    # Create a tower of blocks
    tower_blocks = sim.create_tower(
        base_position=[0.5, 0, 0],
        num_blocks=3,
        block_size=[0.04, 0.04, 0.04],
        spacing=0.01
    )
    
    # Add a sphere nearby
    sphere = sim.add_sphere(
        position=[0.3, 0.2, 0.05],
        radius=0.05,
        rgba_color=[1, 0, 0, 1],
        name="red_sphere"
    )
    
    # Let simulation settle and analyze contacts
    sim.step(500, update_contacts=True)
    
    # Print contact and support info
    #sim.print_contact_info()
    #sim.print_support_info()
    
    # Visualize the graphs
    #sim.visualize_graph(sim.contact_graph, "Contact Graph")
    #sim.visualize_graph(sim.support_graph, "Support Graph")

    print(sim.contact_graph)
    print(sim.support_graph)

    contact = sim.contact_tensor
    sim.build_contact_tensor()
    sim.build_support_tensor()
    print(sim.contact_tensor)
    print(sim.support_tensor)
    from rinarak.utils import stprint
    stprint(sim.get_object_attributes())
    
    # Pick the top block
    sim.pick_object(tower_blocks[-1])
    
    # Re-analyze after picking
    #sim.update_contact_analysis(visualize=True)
    
    # Wait for user to close
    #input("Press Enter to exit...")
    sim.close()