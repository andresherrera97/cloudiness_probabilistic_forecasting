# https://www.goes-r.gov/downloads/resources/documents/Beginners_Guide_to_GOES-R_Series_Data.pdf
# slide 32
# in order to minimize file size, many of the ABI L1b/L2+ products use 16-bit scaled integers for
# physical data quantities rather than 32-bit floating point values.
# unpacked_value = packed_value * scale_factor + add_offset
# The variables ‘scale_factor’ and ‘add_offset’ are included in the metadata for each file.
# The scale factor is calculated with the formula (Max Value - Min Value)/(2^12-1),
# and the offset is the product’s expected minimum value.
# information extracted from https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_MCMIPC
# Max value = 1.3, Min value = 0

CORRECTION_FACTOR = (2**12 - 1) / 1.3

BUCKET = "noaa-goes16"

SENSOR = "ABI"  # Advanced Baseline Imager
PROCESSING_LEVEL = "L2"  # [L1b, L2]
PRODUCT = "CMIP"  # [CMIP, Rad]

REGION = "F"  # [C, F]
# For CONUS there is an image every 5 minutes (60/5 = 12 images per hour)
# Full-Disk has an image every 10 minutes (60/10 = 6 images per hour)

PREFIX = f"{SENSOR}-{PROCESSING_LEVEL}-{PRODUCT}{REGION}"

CHANNEL = "C02"  # Visible Red Band
