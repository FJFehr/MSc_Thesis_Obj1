
# modes of variation - This plots the diagrams
# plot modes of variation -  this fetches the diagrams and then compiles them into the graph so we need a preprocessing step
from src.meshManipulation import trim, PlotModesVaration
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from PIL import Image

def PlotCombineModesVariation(number_of_modes, name1,name2,trim_type):
    names = [name1,name2]

    # # combine and save. Then pull in
    # plt.tight_layout(pad=0)
    min_extreme = -3
    pics_per_mode = 7
    imlist = []

    # This sorta works but the gaps between images is a fuck up
    for i in range(number_of_modes):
        for j in range(pics_per_mode):
            for l in range(len(names)):
                name = names[l]

                if (min_extreme + j == 0):
                    img = Image.open('../results/' + name + 'mean.png')
                    img = trim(img, trim_type=trim_type)
                    imlist.append(img)
                else:
                    img = Image.open('../results/' + name + "mode_" + str(i + 1) + str((min_extreme + j)) + '.png')
                    img = trim(img, trim_type=trim_type)
                    imlist.append(img)

            widths, heights = zip(*(i.size for i in imlist))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in imlist:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]
            if (min_extreme + j == 0):
                new_im.save('../results/' + names[0] + names[1] + "mean.png")
            else:
                new_im.save(
                    '../results/' + names[0] + names[1] + "mode_" + str(i + 1) + str((min_extreme + j)) + '.png')
            imlist = []

    PlotModesVaration(3, names[0] + names[1], trim_type="none")

def main():

    trim_type = "faust"
    number_of_modes = 3

    PlotCombineModesVariation(number_of_modes,"faust_PCA_","faust_linear_AE_",trim_type)
    PlotCombineModesVariation(number_of_modes, "faust_linear_AE_", "faust_nonlinear_AE_", trim_type)

    trim_type = "femur"
    PlotCombineModesVariation(number_of_modes, "femur_PCA_", "femur_linear_AE_", trim_type)
    PlotCombineModesVariation(number_of_modes, "femur_linear_AE_", "femur_nonlinear_AE_", trim_type)




if __name__ == '__main__':
    main()