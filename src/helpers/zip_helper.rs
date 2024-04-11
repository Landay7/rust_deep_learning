// Helper code, might be used in the future

use std::io::prelude::*;

fn list_zip_contents(reader: impl Read + Seek) -> zip::result::ZipResult<()> {
    let mut zip = zip::ZipArchive::new(reader)?;

    for i in 0..zip.len() {
        let mut file = zip.by_index(i)?;
        println!("Filename: {}", file.name());
        
        if file.name() == "config.json" {
            let mut buf = Vec::new();
            let _ = std::io::copy(&mut file, &mut buf);
            let deserialized: Config = serde_json::from_slice(&buf).unwrap();
            println!("{:#?}", deserialized);
        } else {
            let _ = std::io::copy(&mut file, &mut std::io::stdout());
        }
        if file.name() == "model.weights.h5" {
            let mut tmp_file = tempfile::NamedTempFile::new().unwrap();
            let _ = std::io::copy(&mut file, &mut tmp_file);
            let path = tmp_file.into_temp_path();
            let hdf5_file = hdf5::File::open(path).unwrap();
            for group in hdf5_file.groups(){
                println!("{:?}", group);
            }
        }
        println!("");
    }

    Ok(())
}