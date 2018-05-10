import bokeh
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import CustomJS, ColumnDataSource, HoverTool, BoxZoomTool, Jitter, ResetTool, WheelZoomTool, ZoomInTool, ZoomOutTool
from bokeh.models.glyphs import Patch

models = {
    "303": {
        "train_accuracy": [0.037444250706619046, 0.04100652180010587, 0.04242400418902414, 0.04441262933829371, 0.049295190162213044, 0.05943051707329228, 0.06752589248995482, 0.0728168803345319, 0.07692408496653869, 0.08143534838460051, 0.0894280710142309, 0.09870613780377976, 0.11096332003376491, 0.12593097313632548, 0.14187055261768072, 0.15730451201781318, 0.17139524806286976, 0.18489955329696478, 0.19548589731710764, 0.2034207411643369, 0.20851406793838684],
        "train_loss": [9.598957799536581, 9.054054112884028, 7.858809475357141, 7.710723069226572, 7.663340189813567, 7.620200987593851, 7.575901708014813, 7.528195840836867, 7.477377158875726, 7.411948960842754, 7.323771846810082, 7.197408392664857, 7.0265667469266635, 6.812857714821807, 6.564957623527012, 6.304818406800613, 6.058501486797485, 5.832198757296135, 5.638499956835696, 5.50091079640853, 5.452446578490222],
        "val_accuracy": [0.04142350443139743, 0.04142350443139743, 0.04120509514968133, 0.04134924528772219, 0.06360951919022188, 0.07312779579007504, 0.0776619725457872, 0.08052313417797194, 0.08382985073612752, 0.09146543898324719, 0.10287950762935756, 0.1175915563386897, 0.13725276011687218, 0.15942130290150958, 0.1804803255815262, 0.19579955248429284, 0.2125122308666509, 0.22552505569927747, 0.2488730082354517, 0.2548050041403439, 0.2772924238000964],
        "val_loss": [9.579989209400397, 8.169696573541922, 7.75504013273614, 7.696137205095374, 7.651556456125679, 7.597448285378522, 7.54404382696357, 7.48412891640022, 7.410159509042671, 7.32092637735137, 7.185639156820656, 7.01003089016431, 6.775301070929173, 6.5033179174297375, 6.205506778576536, 5.899043588788942, 5.626846039631795, 5.390138840855318, 5.169557060114259, 4.999261070237601, 4.841535982687046],
        "label": "optimizer = SGD",
    },
    "304": {
        "train_accuracy": [0.17881573581512702, 0.3908101707425679, 0.46988996486636125, 0.5155180476264911, 0.5468140496459617, 0.5703739611409426, 0.5876513038897454, 0.6026003921165701, 0.6140789418887717, 0.6240559220346572, 0.6326831251001653, 0.6395029866601074, 0.6468186363655757, 0.6519021355785868, 0.6563063771520173, 0.6614433864973632, 0.6656947409523649, 0.6693083384002316, 0.6724458028913686, 0.6758442644436172, 0.6783101161958248, 0.681050072505067, 0.6835268448693592, 0.685688012899211, 0.6878513677705916, 0.6894785259767547, 0.6919749516226025, 0.6937386154882564],
        "train_loss": [5.961588338410941, 3.4789609537078934, 2.7997097995521307, 2.455918640680698, 2.236598714554538, 2.0846053246058784, 1.9709000393142349, 1.8831813172092857, 1.8099483708361728, 1.748616412961218, 1.698393110655707, 1.6555943459219922, 1.6161528481736627, 1.5863935489966812, 1.5572722479975403, 1.530100953298822, 1.5055556599894881, 1.4832979372585988, 1.4644339570858074, 1.4466895000623188, 1.4316074748226673, 1.4158319567294793, 1.400525153783692, 1.3880912064866784, 1.3751149778739933, 1.3657594710457492, 1.3542791063657886, 1.34253720226468],
        "val_accuracy": [0.3941981754186835, 0.520700829902907, 0.5738136002078948, 0.6045394202192835, 0.6259697368266964, 0.640013453261893, 0.6510212807672664, 0.6624397206463093, 0.6694593956625792, 0.6757408464408168, 0.6793314934834975, 0.6858182496195703, 0.6879324505982001, 0.6914444707911008, 0.6939430724551366, 0.6963979924419988, 0.698263208418543, 0.6998182819655797, 0.7012772560202946, 0.7029764804763103, 0.7053484054319279, 0.7065409203089913, 0.7083100344228584, 0.7084541853526667, 0.709375872462363, 0.7106775924425935, 0.7116429624125582, 0.7114813386652434],
        "val_loss": [3.7125039169466425, 2.685636490225159, 2.310194902707319, 2.1019725140026906, 1.9700426628444994, 1.8826033454518563, 1.8175003047442015, 1.7581149386499793, 1.715223459148327, 1.6820147224385078, 1.6619261377744912, 1.6338231302120847, 1.6203864664714165, 1.6034171987019807, 1.590639027505738, 1.5772739035859267, 1.5715622183894657, 1.564015924055749, 1.5553515120978325, 1.5467524557580887, 1.5367550166385748, 1.5359324799308505, 1.523002839909941, 1.5240014998280271, 1.5188138621177358, 1.5120798346047832, 1.5110754527907682, 1.520249984235713],
        "label": "optimizer = Adam",
    },
    "305": {
        "train_accuracy": [0.06739047802522728, 0.12873972897702995, 0.1779060573368309, 0.21281347309711207, 0.23855202794049302, 0.2547012810175385, 0.2692190093485012, 0.2798599558323749, 0.28856469537444557, 0.29609766391132103, 0.30201221348680196],
        "train_loss": [7.408853589584706, 6.0881027109241455, 5.27752906277486, 4.814890250500027, 4.519619375770496, 4.320367657745696, 4.170717443167275, 4.05749119266451, 3.970362289261073, 3.8968321430337736, 3.8355405535247797],
        "val_accuracy": [0.10972882253772676, 0.1942139013243648, 0.2580243567322965, 0.3035452187421654, 0.3347034874427642, 0.3621881122083547, 0.3785600715199929, 0.3912409135552635, 0.40140568174636326, 0.4114131958963053, 0.42103630881095216],
        "val_loss": [6.610289281761643, 5.2795244225717175, 4.5631917496783325, 4.140626254730262, 3.865115000960345, 3.6896562001409894, 3.541056814492332, 3.4209903692246018, 3.3356743636563198, 3.2762990669894894, 3.2148012079804644],
        "label": "kernel size = 8x8",
    },
    "306": {
        "train_accuracy": [0.2348, 0.5244, 0.6356, 0.7055, 0.7540, 0.7872, 0.8135, 0.8368, 0.8551, 0.8712, 0.8852, 0.8976, 0.9082, 0.9166],
        "train_loss": [5.4184, 2.5415, 1.8139, 1.4291, 1.1814, 1.0091, 0.8760, 0.7636, 0.6705, 0.5931, 0.5264, 0.4700, 0.4238, 0.3880],
        "val_accuracy": [0.4421, 0.5591, 0.6006, 0.6302, 0.6381, 0.6474, 0.6506, 0.6496, 0.6440, 0.6469, 0.6492, 0.6502, 0.6506, 0.6503],
        "val_loss": [3.2465, 2.4024, 2.1744, 2.0712, 2.0995, 2.0866, 2.1405, 2.2612, 2.4392, 2.5002, 2.5620, 2.7389, 2.8245, 2.9149],
        "label": "no dropout layers",
    }
}

output_file("comparison_cnn1_vs_cnn2.html")
p1 = figure(title="Comparison CNN1 (kernel size 3x3) vs CNN2 (kernel size 8x8)", x_axis_label='epoch', y_axis_label='accuracy', y_range=(0, 1), plot_width=600, plot_height=400)
p1.title.align = "center"
p1.title.text_font_size = "18px"

# data = models['304']['val_accuracy']
# p1.line([i for i in range(0, len(data) + 1)], [0] + data, legend="CNN1: Kernel Size 3x3", line_width=2, color='darkblue')
#
# data = models['305']['val_accuracy']
# p1.line([i for i in range(0, len(data) + 1)], [0] + data, legend="CNN2: Kernel Size 8x8", line_width=2, color='orangered')
#
# p1.xaxis.axis_label_text_font_size = '12pt'
# p1.yaxis.axis_label_text_font_size = '12pt'
# p1.legend.location = 'bottom_right'

data1 = [0] + models['304']['train_accuracy']
data2 = [0] + models['304']['val_accuracy']
x = np.hstack(([i for i in range(0, len(data2))], [i for i in range(0, len(data1))][::-1]))
y = np.hstack((data2, data1[::-1]))
source = ColumnDataSource(dict(x=x, y=y))
p1.line([i for i in range(0, len(data1) + 1)], data1, legend="Training kernel size 3x3", line_width=2, color='forestgreen', line_dash='dashed')
p1.line([i for i in range(0, len(data2) + 1)], data2, legend="Validation kernel size 3x3", line_width=2, color='forestgreen')
glyph = Patch(x='x', y='y', fill_color="forestgreen", fill_alpha=0.1, line_alpha=0)
p1.add_glyph(source, glyph)

data3 = [0] + models['305']['train_accuracy']
data4 = [0] + models['305']['val_accuracy']
x = np.hstack(([i for i in range(0, len(data4))], [i for i in range(0, len(data3))][::-1]))
y = np.hstack((data4, data3[::-1]))
source = ColumnDataSource(dict(x=x, y=y))
p1.line([i for i in range(0, len(data3) + 1)], data3, legend="Training kernel size 8x8", line_width=2, color='salmon', line_dash='dashed')
p1.line([i for i in range(0, len(data4) + 1)], data4, legend="Validation kernel size 8x8", line_width=2, color='salmon')
glyph = Patch(x='x', y='y', fill_color="salmon", fill_alpha=0.3, line_alpha=0)
p1.add_glyph(source, glyph)

p1.xaxis.axis_label_text_font_size = '12pt'
p1.yaxis.axis_label_text_font_size = '12pt'
p1.legend.location = 'bottom_right'

show(p1)


output_file("comparison_dropout_vs_without_dropout.html")
p2 = figure(title="Comparison CNN with vs without Dropout Layer", x_axis_label='epoch', y_axis_label='accuracy', y_range=(0, 1), plot_width=600, plot_height=400)
p2.title.align = "center"
p2.title.text_font_size = "20px"

data1 = [0] + models['304']['train_accuracy']
data2 = [0] + models['304']['val_accuracy']
x = np.hstack(([i for i in range(0, len(data2))], [i for i in range(0, len(data1))][::-1]))
y = np.hstack((data2, data1[::-1]))
source = ColumnDataSource(dict(x=x, y=y))
p2.line([i for i in range(0, len(data1) + 1)], data1, legend="Training With Dropout", line_width=2, color='darkblue', line_dash='dashed')
p2.line([i for i in range(0, len(data2) + 1)], data2, legend="Validation With Dropout", line_width=2, color='darkblue')
glyph = Patch(x='x', y='y', fill_color="darkblue", fill_alpha=0.1, line_alpha=0)
p2.add_glyph(source, glyph)

data3 = [0] + models['306']['train_accuracy']
data4 = [0] + models['306']['val_accuracy']
x = np.hstack(([i for i in range(0, len(data4))], [i for i in range(0, len(data3))][::-1]))
y = np.hstack((data4, data3[::-1]))
source = ColumnDataSource(dict(x=x, y=y))
p2.line([i for i in range(0, len(data3) + 1)], data3, legend="Training Without Dropout", line_width=2, color='orangered', line_dash='dashed')
p2.line([i for i in range(0, len(data4) + 1)], data4, legend="Validation Without Dropout", line_width=2, color='orangered')
glyph = Patch(x='x', y='y', fill_color="orangered", fill_alpha=0.3, line_alpha=0)
p2.add_glyph(source, glyph)

p2.xaxis.axis_label_text_font_size = '12pt'
p2.yaxis.axis_label_text_font_size = '12pt'
p2.legend.location = 'bottom_right'

show(p2)


output_file("comparison_adam_vs_sgd.html")
p3 = figure(title="Comparison Adam vs SGD", x_axis_label='epoch', y_axis_label='accuracy', y_range=(0, 1), plot_width=600, plot_height=400)
p3.title.align = "center"
p3.title.text_font_size = "20px"

data1 = [0] + models['304']['train_accuracy']
data2 = [0] + models['304']['val_accuracy']
x = np.hstack(([i for i in range(0, len(data2))], [i for i in range(0, len(data1))][::-1]))
y = np.hstack((data2, data1[::-1]))
source = ColumnDataSource(dict(x=x, y=y))
p3.line([i for i in range(0, len(data1) + 1)], data1, legend="Training Adam", line_width=2, color='darkorchid', line_dash='dashed')
p3.line([i for i in range(0, len(data2) + 1)], data2, legend="Validation Adam", line_width=2, color='darkorchid')
glyph = Patch(x='x', y='y', fill_color="darkorchid", fill_alpha=0.1, line_alpha=0)
p3.add_glyph(source, glyph)

data3 = [0] + models['303']['train_accuracy']
data4 = [0] + models['303']['val_accuracy']
x = np.hstack(([i for i in range(0, len(data4))], [i for i in range(0, len(data3))][::-1]))
y = np.hstack((data4, data3[::-1]))
source = ColumnDataSource(dict(x=x, y=y))
p3.line([i for i in range(0, len(data3) + 1)], data3, legend="Training SGD", line_width=2, color='goldenrod', line_dash='dashed')
p3.line([i for i in range(0, len(data4) + 1)], data4, legend="Validation SGD", line_width=2, color='goldenrod')
glyph = Patch(x='x', y='y', fill_color="goldenrod", fill_alpha=0.3, line_alpha=0)
p3.add_glyph(source, glyph)

p3.xaxis.axis_label_text_font_size = '12pt'
p3.yaxis.axis_label_text_font_size = '12pt'
p3.legend.location = 'bottom_right'
p3.legend.label_text_font_size = '8pt'

show(p3)
