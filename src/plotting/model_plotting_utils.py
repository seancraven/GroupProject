import torch
import matplotlib.figure as fig
import matplotlib.pyplot as plt

from src.testing.model_testing_utils import ModelMetrics

from typing import Dict


def plot_bar(
        title : str, 
        x_label : str, 
        y_label : str,
        **entries : float
        ) -> fig.Figure:
    """ 
    Plots a barplot for names entries

    Arguments:
    ----------
    title       : {String}
                    > Plot title.
    x_label     : {String}
                    > Plot x-axis label.
    y_label     : {String}
                    > Plot y-axis label.
    entries     : {Dictionary}
                    > Dictionary of names entries to plot

    Returns:
    ----------
    plt.gcf()   : {Figure}
                    > Matplotlib figure object
    """
    # Create a new figure
    fig = plt.figure()

    # Get the names and values
    names = list(entries.keys())
    values = list(entries.values())

    # Plot the barplot
    plt.bar(names, values)

    # Add plot labels
    #  if title not in [None, ''] :
    plt.title(title)

    # if x_label not in [None, ''] :
    plt.xlabel(x_label)
    
    # if y_label not in [None, ''] :
    plt.ylabel(y_label)

    return fig


def plot_img(
        image : torch.Tensor,
        title : str = None,
        x_label : str = None,
        y_label : str = None,
        ) -> fig.Figure :
    """
    Plots image

    Arguments:
    ----------
    title   : {String}
                > Plot title.
    x_label : {String}
                > Plot x-axis label.
    y_label : {String}
                > Plot y-axis label.
    image   : {Tensor}
                > Image tensor to plot

    Returns:
    ----------
    fig     : {Figure}
                > Matplotlib figure object
    """
    # Create a new figure
    fig = plt.figure()

    # Plot the image
    img = image.squeeze().permute(1, 2, 0)
    plt.imshow(img)

    # Add plot title
    if title not in [None, ''] :
        plt.title(title)

    if x_label not in [None, ''] :
        plt.xlabel(x_label)

    if y_label not in [None, ''] :
        plt.ylabel(y_label)

    # Return the figure object
    return fig


def plot_img_row(
        title : str = None,
        x_label : str = None,
        y_label : str = None,
        **images : torch.Tensor
        ) -> fig.Figure :
    """
    Plots a row of images by names arguments in a row
    (i.e image name is argument name)

    Arguments:
    ----------
    title   : {String}
                > Plot title.
    x_label : {String}
                > Plot x-axis label.
    y_label : {String}
                > Plot y-axis label.
    images  : {Dictionary}
                > Dictionary of images to plot

    Returns:
    ----------
    fig     : {Figure}
                > Matplotlib figure object
    """

    # Get the names and values
    keys = list(images.keys())
    values = list(images.values())

    fig, axs = plt.subplots(1, len(keys), figsize=(20, 8))

    for i, ax in enumerate(axs.flat):
        
        img = values[i].squeeze().permute(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"Image Name: \n{keys[i]}")

        if x_label not in [None, ''] :
            ax.set_xlabel(x_label)
    
        if y_label not in [None, ''] :
            ax.set_ylabel(y_label)
    
        ax.set_xticks([])
        ax.set_yticks([])

    # Add plot title 
    if title not in [None, ''] :
        plt.suptitle(title, fontsize=16, fontweight='bold')
    
    return fig


def plot_img_label_grid(
        title : str = None,
        x_label : str = None,
        y_label : str = None,
        **img_label_dict : Dict[str, torch.Tensor]
        ) -> fig.Figure :
    """
    Plots a grid of images by names arguments in a row with labels
    (i.e image name is argument name)

    Arguments:
    ----------
    title   : {String}
                > Plot title.
    x_label : {String}
                > Plot x-axis label.
    y_label : {String}
                > Plot y-axis label.
    images  : {Ordered Dictionary}
                > {'image_dictionary' : {'img_name' : torch.Tensor}} or,
                > {'label_dictionary' : {'label_name' : torch.Tensor}}

    Returns:
    ----------
    fig     : {Figure}
                > Matplotlib figure object
    """
    # Get the dictionary names
    dictionary_names = list(img_label_dict.keys())

    assert dictionary_names == ['image_dictionary', 'label_dictionary'], "Dictionary names must be 'image_dictionary' and 'label_dictionary'"

    # get image dictionary
    img_dict = img_label_dict['image_dictionary']
    img_dict_values = list(img_dict.values())
    img_dict_keys = list(img_dict.keys())

    # get label dictionary
    label_dict = img_label_dict['label_dictionary']
    label_dict_values = list(label_dict.values())
    label_dict_keys = list(label_dict.keys())

    assert len(img_dict) == len(label_dict), "Image and label dictionaries must be the same length"

    # plot
    fig, axs = plt.subplots(2, len(img_dict), figsize=(20, 8))

    for i, ax in enumerate(axs.flat):
        if i < len(label_dict):
            # plot the images
            img = img_dict_values[i].squeeze().permute(1, 2, 0)
            ax.imshow(img)
            ax.set_title(f"Image Name: \n{img_dict_keys[i]}")

        else:
            # get weird label
            label_idx = i - len(label_dict)
            seg_label = label_dict_values[label_idx].squeeze()
            ax.imshow(seg_label)
            ax.set_title(f"Image Label: \n{label_dict_keys[label_idx]}")
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add plot title
    if title not in [None, ''] :
       plt.suptitle(title, fontsize=16, fontweight='bold')
    
    if x_label not in [None, ''] :
        plt.xlabel(x_label)
    
    if y_label not in [None, ''] :
        plt.ylabel(y_label)

    return fig
