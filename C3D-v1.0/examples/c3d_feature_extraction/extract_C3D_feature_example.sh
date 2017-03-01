VIDEOFILE=input/avi/v_BaseballPitch_g01_c01.avi
OUTDIR=.

# remove already extracted feature
rm -f \
  v_BaseballPitch_g01_c01_0000*.csv

# run extraction
python \
  extract_C3D_feature.py \
  ${VIDEOFILE} \
  ${OUTDIR}
