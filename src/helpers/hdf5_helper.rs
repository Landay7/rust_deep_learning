// helps to debug hdf5 File

pub fn traverse_groups(group: &hdf5::Group) {
    println!("Group: {}", group.name());

     for dataset in group.datasets().expect("Failed to list datasets") {
        // Check the dimensionality of the dataset
        println!("{:?}", dataset.name());
        let dimensions = dataset.shape();
        match dimensions.len() {
            1 => {
                // 1D array
                println!("1D Array:");
            }
            2 => {
                // 2D array
                println!("2D Array:");
            }
            _ => {
                println!("Unsupported array dimensionality {}", dimensions.len() );
            }
        }
    }

    // Recursively traverse subgroups
    for subgroup in group.groups().expect("Failed to list subgroups") {
        traverse_groups(&subgroup);
    }
}