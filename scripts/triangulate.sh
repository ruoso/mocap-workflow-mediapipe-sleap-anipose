#!/bin/sh
session="$1"

slap-triangulate \
	--p2d session/${session}\
       	--calib session/${session}/calibration.toml \
	--fname session/${session}/points3d.h5 \
	--frames 1 509

