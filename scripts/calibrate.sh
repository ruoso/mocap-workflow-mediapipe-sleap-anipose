#!/bin/bash
session="$1"
slap-calibrate \
	--session session/${session}/ \
	--board board.toml \
	--calib_fname session/${session}/calibration.toml \
	--metadata_fname session/${session}/calibration_metadata.h5 \
	--histogram_path session/${session}/reprojection_histogram.png \
	--reproj_path session/${session}


