# Data for stars-in-Planck statistical methods project

The data are cutouts from Planck maps at 857 GHz.  You can load them using 'pickle' in python.

File descriptions:

-generate_cutouts.ipynb: file used to make the cutouts.  You can ignore this unless you're curious.

-cutout_data/cutouts_1000offstarfirstpass_01.pk: python dictionary containing cutouts of Planck 857 GHz maps centered at random (off star) locations.  'cutouts' is the cutouts, 'l' is galactic longitude, 'b' is galactic latitude.  Cutouts are in units of MJy/str.

-cutout_data/cutouts_1000onstar_firstpass_01.pk: same as above, but with cutouts centered on locations of real stars.

-ptsource_template.pk: a matrix with the same dimension as the cutouts representing a point source observed by Planck with flux of 1 Jy (reasonable for a large, nearby debris disk).  This template can be used to quickly generate a training set for which we know the amplitude of the point source at the image center.
