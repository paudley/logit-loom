// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inspects the versioned companion ABI without loading a model.

use std::io::{self, Write as _};

use logit_loom_diffusion_sdcpp::probe_companion;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let library = arguments
        .next()
        .ok_or("usage: probe_companion COMPANION_LIBRARY")?;
    if arguments.next().is_some() {
        return Err("usage: probe_companion COMPANION_LIBRARY".into());
    }

    let receipt = probe_companion(library)?;
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &receipt)?;
    writeln!(output)?;
    Ok(())
}
