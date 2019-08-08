import tusimple_data_parser as tdp

tdp.pre_processing(data_location='./label_data_0313.json', processed_images_location='./image/image{0:04d}.png',
                   processed_labels_location='./label/label{0:04d}.png',
                   line_weights=10,
                   plot_images=False)
