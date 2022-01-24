from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import aplpy
import AG_fft_tools as fft_tools
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
import aplpy
import pyparsing
import pyregion
import matplotlib.pyplot as pyplot
import matplotlib as mpl
import os
import numpy as np
import matplotlib.text as text

def combine_2d(
               fits_im_lowres,
               fits_im_highres,
               fits_output,
               lowresfwhm,
               pixscale,
               angular_scales,
               kernel,
               verbose=False,
               generatefig=True
               ):
    """
        Purpose:
        Combine input high and low angular resolution fits images in Fourier domain,
        and the output the combined image
        
        Input:
        fits_im_lowres  [string]: filename of the input low resolution FITS image
        fits_im_highres [string]: filename of the input high resolution image
        fits_output     [string]: output filename
        kernel          [string]: default: setup function. Options: step/feather/immerge
        generatefig     [bool]  : default: True. If False, then it will only produce combined images in FITS
        """
    
    # Read input images
    hdu_low = fits.open(fits_im_lowres)
    header_low = hdu_low[0].header
    data_low = hdu_low[0].data
    
    hdu_high = fits.open(fits_im_highres)
    header_high = hdu_high[0].header
    data_high = hdu_high[0].data
    
    # sanity check
    assert header_low['NAXIS'] == data_low.ndim == 2, 'Error: Input lores image dimension non-equal to 2.'
    assert header_high['NAXIS'] == data_high.ndim == 2, 'Error: Input hires image dimension non-equal to 2.'
    
    # regrid input images
    if ( header_low['CDELT1'] == header_low['CDELT1'] ):
        if ( header_low['CDELT2'] == header_low['CDELT2'] ):
            print ('input images have the same pixelscale')
        else:
            hdu_low, data_low, nax1, nax2, pixscale = regrid(header_high,data_high,data_low,header_low)
            print ('regrid input low resolution image')
    else:
        hdu_low, data_low, nax1, nax2, pixscale = regrid(header_high,data_high,data_low,header_low)
        print ('regrid input low resolution image')
    
    # give input images the same x and y dimension
    data_low_same, header_low_same = same_dimension(data_low,header_low)
    data_high_same, header_high_same = same_dimension(data_high,header_high)
    
    # Construct weighting kernals
    # make_kernel(kernel)
    
    if ( verbose == True ):
        print ( 'Construct weighting kernels' )
    if ( kernel == 'step' ):
        if ( verbose == True ):
            print ( 'Use step function as weighting function' )
            nax2 = header_high_same['naxis2']
            nax1 = header_high_same['naxis1']
            kfft, ikfft = step_2d(nax2, nax1, lowresfwhm, pixscale,data_low_same)

    if ( kernel == 'step_new' ):
        if ( verbose == True ):
            print ( 'Use new step function as weighting function' )
            nax2 = header_high_same['naxis2']
            nax1 = header_high_same['naxis1']
            kfft, ikfft = step_2d_new(nax2, nax1, lowresfwhm, pixscale,data_low_same)

    if ( kernel == 'step_butter' ):
        if ( verbose == True ):
            print ( 'Use butterworth function as weighting function' )
            nax2 = header_high_same['naxis2']
            nax1 = header_high_same['naxis1']
            kfft, ikfft = butterworth(nax2, nax1, lowresfwhm, pixscale,data_low_same)

    if ( kernel == 'feather' ):
        if ( verbose == True ):
            print ( 'Use feather function as weighting function' )
            nax2 = header_high_same['naxis2']
            nax1 = header_high_same['naxis1']
            kfft, ikfft = feather_2d(nax2, nax1, lowresfwhm, pixscale)

    # Combine and inverse fourier transform the images
    combo_im = fft2_add(data_low_same,data_high_same,kfft,ikfft)
    
    # output combined image
    outpath = fits_output
    header = header_high_same
    if os.path.exists(outpath):
        os.remove(outpath)
    fits.writeto(outpath,combo_im,header=header)
    # Generate figures
    plot_ps(
            data_low_same,
            data_high_same,
            combo_im,
            pixscale,
            angular_scales,
            linestyle_dict,
            color_dict,
            )

def fft2_add(
             data_low_same,
             data_high_same,
             kfft,
             ikfft,
             ):
    """
        Purpose:
        Combine and inverse fourier transform the 1d array
        
        Input:
        data_low        [float array]: The low resolution 1d array
        data_high       [float array]: The high resolution 1d array
        kfft            [float array]: Weighting function for low resolution data
        ikfft           [float array]: Weighting function for high resolution data
        
        return:
        combo_im        [float array]: The combined 1d array
        """
    
    # Combine and inverse fourier transform the images
    fft_high = np.fft.fft2(np.nan_to_num(data_high_same))
    fft_low = np.fft.fft2(np.nan_to_num(data_low_same))
    
    fftsum = kfft*fft_low + ikfft*fft_high
    combo = np.fft.ifft2(fftsum)
    combo_im = combo.real
    
    return combo_im

def step_2d(
            nax2,
            nax1,
            lowresfwhm,
            pixscale,
            data_low_same,
            ):
    """
        Purpose:
        Construct the weight kernels (image arrays) for the fourier transformed low
        resolution and high resolution images.  The kernels are the step fuctions in
        fourier transforms with the step at low-resolution beam and (1-[that kernel])
        Parameters.
        Input:
        nax2,nax1  [int] : Number of pixels in each axes
        lowresfwhm [float] : Angular resolution of the low resolution image (FWHM)
        pixscale [float] : The pixel size in the input high resolution image
        Output:
        kfft  [float array]: An image array containing the weighting for the low resolution image
        ikfft [float array]: An image array containing the weighting for the high resolution image
        """
    ygrid, xgrid = np.indices(data_low_same.shape, dtype='float')
    rr = ((xgrid - data_low_same.shape[1]/2)**2+(ygrid - data_low_same.shape[0]/2)**2)**0.5
    a = 2* lowresfwhm / pixscale
    b = (2 * (data_low_same.shape[1]/2.)**2)**0.5 - a
    ring = (rr >= b)
    iring = 1-ring
    
    kfft = ring
    ikfft = iring
    
    return kfft, ikfft

def step_2d_new(
                nax2,
                nax1,
                lowresfwhm,
                #highresfwhm,
                pixscale,
                data_low_same,
                ):
    """
        Purpose:
        Construct the weight kernels for the fourier transformed low
        resolution and high resolution 1d arrays.  The kernels are the step fuctions in
        fourier transforms with the step at low-resolution beam and (1-[that kernel])
        Parameters.
        Input:
        x           [int] : Number of pixels
        lowresfwhm  [float] : Angular resolution of the low resolution image (FWHM)
        highresfwhm [float] : Angular resolution of the high resolution image (FWHM)
        pixscale    [float] : The pixel size in the input high resolution image
        Output:
        kfft  [float array]: 1d array containing the weighting for the low resolution 1d array
        ikfft [float array]: 1d array containing the weighting for the high resolution 1d array
    """
    ygrid, xgrid = np.indices(data_low_same.shape, dtype='float')
    fwhm = np.sqrt(8*np.log(2))
    sigma_low = lowresfwhm/fwhm/pixscale
    gaussian_low = np.exp(-((xgrid - data_low_same.shape[1]/2)**2+(ygrid - data_low_same.shape[0]/2)**2)/(2*sigma_low**2))
    G_low = np.abs(np.fft.fft2(gaussian_low))
    G_low /= G_low.max()
    
    sigma_high = 14.0/fwhm/pixscale
    gaussian_high = np.exp(-((xgrid - data_low_same.shape[1]/2)**2+(ygrid - data_low_same.shape[0]/2)**2)/(2*sigma_high**2))
    G_high = np.abs(np.fft.fft2(gaussian_high))
    G_high /= G_high.max()
    
    rr = ((xgrid - data_low_same.shape[1]/2)**2+(ygrid - data_low_same.shape[0]/2)**2)**0.5
    a = 2* lowresfwhm / pixscale
    b = (2 * (data_low_same.shape[1]/2.)**2)**0.5 - a
    ring = (rr >= b)
    iring = 1-ring
    
    #x = np.zeros((1711,1711))
    #x[855][855] = 1.

#######
#data_high = gauss_2d(x,14,4.)
#data_low = gauss_2d(x,40,4.)
#kfft = ring*np.fft.fft2(data_high)/np.fft.fft2(data_low)
#ikfft = iring
######
    kfft = ring*G_high/G_low
    ikfft = iring
    
    return kfft, ikfft



def butterworth(
                nax2,
                nax1,
                lowresfwhm,
                #highresfwhm,
                pixscale,
                data_low_same,
                ):
    """
        Purpose:
        Construct the weight kernels for the fourier transformed low
        resolution and high resolution 1d arrays.  The kernels are the step fuctions in
        fourier transforms with the step at low-resolution beam and (1-[that kernel])
        Parameters.
        Input:
        x           [int] : Number of pixels
        lowresfwhm  [float] : Angular resolution of the low resolution image (FWHM)
        highresfwhm [float] : Angular resolution of the high resolution image (FWHM)
        pixscale    [float] : The pixel size in the input high resolution image
        Output:
        kfft  [float array]: 1d array containing the weighting for the low resolution 1d array
        ikfft [float array]: 1d array containing the weighting for the high resolution 1d array
        """
    ygrid, xgrid = np.indices(data_low_same.shape, dtype='float')
    fwhm = np.sqrt(8*np.log(2))
    sigma_low = lowresfwhm/fwhm/pixscale
    gaussian_low = np.exp(-((xgrid - data_low_same.shape[1]/2)**2+(ygrid - data_low_same.shape[0]/2)**2)/(2*sigma_low**2))
    G_low = np.abs(np.fft.fft2(gaussian_low))
    G_low /= G_low.max()
    
    sigma_high = 14.0/fwhm/pixscale
    gaussian_high = np.exp(-((xgrid - data_low_same.shape[1]/2)**2+(ygrid - data_low_same.shape[0]/2)**2)/(2*sigma_high**2))
    G_high = np.abs(np.fft.fft2(gaussian_high))
    G_high /= G_high.max()
    
    rr = ((xgrid - data_low_same.shape[1]/2)**2+(ygrid - data_low_same.shape[0]/2)**2)**0.5
    a1 = 150./fwhm/ pixscale
    b1 = (2 * (data_low_same.shape[1]/2.)**2)**0.5 - a1
    ring1 = (rr <= 500)
    ring1 = np.fft.fftshift(ring1)
    a2 = 40.0 /fwhm/ pixscale
    b2 = (2 * (data_low_same.shape[1]/2.)**2)**0.5 - a2
    
    ring2 = (rr >= 1700)
    ring2 = np.fft.fftshift(ring2)
    #######
    x = np.zeros((1711,1711))
    x[855][855] = 1.
    
    
    data_high = gauss_2d(x,14,4.)
    data_low = gauss_2d(x,40,4.)
    #######
    
    step = 1 - ring1 - ring2
    w_step = step * (np.fft.fft2(data_low)/np.fft.fft2(data_high))**0.85
    w_step = w_step / w_step.max()
    w_step = (w_step - w_step.min())/(w_step.max() - w_step.min())
    w_low = ring1 + w_step
    w_high = 1-w_low
    
    kfft = w_low*np.fft.fft2(data_high)/np.fft.fft2(data_low)
    ikfft = w_high
    #ikfft = 1-kfft

    return kfft, ikfft






angular_scales = [14.0, 40.0, 150.0]
linestyle_dict = {
    14.0: ':',
        40.0: ':',
            150.0: ':'
            }
color_dict      = {
    14.0: (0,0,0),
        40.0: (0,0,1),
            150.0: (1,0,0)
            }

def plot_ps(
            data_low_same,
            data_high_same,
            combo_im,
            pixscale,
            #model_file = False,
            angular_scales,
            linestyle_dict,
            color_dict,
            ):
    
    plt.figure(figsize=(10, 8))
    # fontsize of the tick labels
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    
    #if ( model_file == True ):
    #import model image.
    #hdu_mod=fits.open(model_file)
    #data_mod = hdu_mod[0].data
    #caculate the PSD
    #frequency_mod,zz_mod = fft_tools.PSD2(data_mod,fft_pad=True,oned=True,view=False,wavnum_scale=False)
    #plot PSD
    #plt.plot((frequency_mod),(zz_mod),linewidth=12.,color = (0, 0, 1.0, 0.2),label='original model image')
    model_file='/Users/shjiao/desktop/Oph_COS/model_new/flux850.model.fits'
    hdu_mod=fits.open(model_file)
    data_mod = hdu_mod[0].data
    header_mod = hdu_mod[0].header
    data_mod_same, header_mod_same = same_dimension(data_mod,header_mod)
    frequency_mod,zz_mod = fft_tools.PSD2(data_mod_same,fft_pad=True,oned=True,view=False,wavnum_scale=False)
    #plt.plot((frequency_mod),(zz_mod),linewidth=12.,color = (0, 0, 1.0, 0.2),label='original model image')
    
    #caculate the PSD
    frequency_low,zz_low = fft_tools.PSD2(data_low_same,fft_pad=True,oned=True,view=False,wavnum_scale=False)
    frequency_high,zz_high = fft_tools.PSD2(data_high_same,fft_pad=True,oned=True,view=False,wavnum_scale=False)
    frequency_com,zz_com = fft_tools.PSD2(combo_im,fft_pad=True,oned=True,view=False,wavnum_scale=False)
    #plot PSD
    plt.plot((frequency_low),(zz_low),linewidth=7.5,color=(1.0, 0, 0, 0.4),label='low resolution image',linestyle='dashed')
    plt.plot((frequency_high),(zz_high),linewidth=7.5,color=(0.2, 0.6, 0, 0.4),label='high resolution image',linestyle='dashed')
    plt.plot((frequency_com),(zz_com),linewidth=3.5,color='blue',label='combined image',linestyle='dotted')
    
    #plt.text(0.7, 10000, 'immerge', fontsize=20)
    #add beam/filtering info
    for i in range(len(angular_scales)):
        angular_scale = angular_scales[i]
        plt.axvline(x= 1. / (2 * angular_scale / pixscale) ,
                    linestyle=linestyle_dict[angular_scale],
                    color=color_dict[angular_scale],
                    alpha=0.3,
                    label='%s arcsecond'%str(int(angular_scale)),
                    linewidth=3.
                    )
        i = i + 1
    
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('spatial frequency ($pixel^{-1}$)',size=20)
    plt.ylabel('PSD ($pW^{2}$ $arcsec^{2}$)',size=20)
    plt.title('Power Spectrum',size=20)
    legend = plt.legend(loc='lower left', shadow=True, fontsize='large',frameon=False)
    #plt.ylim((10**-1))

def same_dimension(
                   image,
                   header,
                   ):
    """
        Purpose:
        give input image the same x and y dimension.
        Input:
        image  [float array]: The input 2d image.
        header [header object] : The header of the input image
        Output:
        data_same  [float array]: The new image with the same x and y dimension
        header [float array]: The new header with the same x and y dimension
        """
    l = header['naxis1']
    k = header['naxis2']
    
    if ( l > k ):
        c = l
    else:
        c = k

    data_same = np.zeros([c,c])

    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            data_same[i][j]=image[i][j]+data_same[i][j]
    
    header['naxis1'] = c
    header['naxis2'] = c
    
    return data_same,header

def feather_2d(
               nax2,
               nax1,
               lowresfwhm,
               pixscale,
               ):
    """
        Purpose:
        Construct the weight kernels (image arrays) for the fourier transformed low
        resolution and high resolution images.  The kernels are the fourier transforms
        of the low-resolution beam and (1-[that kernel])
        Parameters
        Input:
        nax2,nax1  [int] : Number of pixels in each axes
        lowresfwhm [float] : Angular resolution of the low resolution image (FWHM)
        pixscale [float] : The pixel size in the input high resolution image
        Output:
        kfft  [float array]: An image array containing the weighting for the low resolution image
        ikfft [float array]: An image array containing the weighting for the high resolution image
        """
    x = np.linspace(0, nax2-1, nax2)
    y = np.linspace(0, nax1-1, nax1)
    x, y = np.meshgrid(x, y)
    fwhm = np.sqrt(8*np.log(2))
    sigma = lowresfwhm/fwhm/pixscale
    
    data = twoD_Gaussian((x, y), 1, nax2 / 2, nax1 / 2, sigma, sigma, 0, )
    #fft_data = np.abs(np.fft.fft2(data.reshape(nax2,  nax1)))
    #fft_data /= fft_data.max()
    #ikfft = np.fft.fftshift(fft_data)
    #kfft = 1 - ikfft
    
    kernel = np.fft.fftshift(data.reshape(nax2,  nax1))
    kfft = np.abs(np.fft.fft2(kernel)) # should be mostly real
    
    # normalize the kernel
    kfft/=kfft.max()
    ikfft = 1-kfft
    
    return kfft, ikfft

def regrid(
           hd1,
           im1,
           im2raw,
           hd2
           ):
    """
        Purpose:
        Regrid the low resolution image to have the same dimension and pixel size with the
        high resolution image.
        Parameters
        Input:
        hd1  [header object] : The header of the high resolution image
        im1  [float array] : The high resolution image
        im2raw [float array] : The pre-regridded low resolution image
        hd2  [header object] : The header of the low resolution image
        Output:
        hdu2  [image and the header]: This will containt the regridded low resolution image,
        and the image header taken from the high resolution observation.
        im2 [float array]: The image array which stores the regridded low resolution image.
        nax1, nax2 [int array] : Number of pixels in each of the spatial axes.
        pixscale [float value] : Pixel size in the input high resolution image.
        """
    
    # Sanity Checks:
    assert hd2['NAXIS'] == im2raw.ndim == 2, 'Error: Input lores image dimension non-equal to 2.'
    assert hd1['NAXIS'] == im1.ndim == 2, 'Error: Input hires image dimension non-equal to 2.'
    
    # read pixel scale from the header of high resolution image
    pixscale = FITS_tools.header_tools.header_to_platescale(hd1)
    log.debug('pixscale = {0}'.format(pixscale))
    
    # read the image array size from the high resolution image
    nax1,nax2 = (hd1['NAXIS1'],
                 hd1['NAXIS2'],
                 )
        
    # create a new HDU object to store the regridded image
    hdu2 = fits.PrimaryHDU(data=im2raw, header=hd2)
                 
    # regrid the image
    hdu2 = hcongrid_hdu(hdu2, hd1)
    im2 = hdu2.data.squeeze()
                 
    # return variables
    return hdu2, im2, nax1, nax2, pixscale

def twoD_Gaussian(
                  (x, y),
                  amplitude,
                  xo,
                  yo,
                  sigma_x,
                  sigma_y,
                  theta,
                  ):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    #h = amplitude*np.exp (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2))
    return g.ravel()

def gauss_2d(
             data_2d,
             fwhm,
             pixel,
             ):
    
    FWHM_TO_SIGMA = 1./np.sqrt(8*np.log(2))
    kernel_size=fwhm*FWHM_TO_SIGMA
    pixel_n = kernel_size/pixel
    gauss_2D_kernel =Gaussian2DKernel(pixel_n)
    smoothed_data_gauss = convolve(data_2d, gauss_2D_kernel,normalize_kernel=True)
    
    return smoothed_data_gauss



