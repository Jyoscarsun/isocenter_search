# class to define constants for the models and outputs
# not super relevant right now - didn't really implement it correctly
class ModelParameters:

    def __init__(self,
                 output_directory=r'D:\GK - Full Pipeline Output',
                 prediction_name='baseline',
                 # raw_data_directory=r'D:\GK - Full Pipeline Test\C',
                 raw_data_directory=r'D:\Gamma Knife\Anonymized Data\C',
                 kernel_data_directory=r'D:\Gamma Knife Downloaded Kernels'):

        self.output_directory = output_directory
        self.data_directory = raw_data_directory
        self.kernel_directory = kernel_data_directory

        self.patient_number = None
        self.prediction_name = prediction_name

        self.patient_shape = (512, 512, 256)

        # creating and identifying appropriate directory paths
        self.main_data_dir = f'{self.output_directory}/all-data'
        self.training_data_dir = f'{self.main_data_dir}/pred-train'
        self.testing_data_dir = f'{self.main_data_dir}/pred-test'

        self.prediction_dir = f'{self.output_directory}/{self.prediction_name}/predictions'

    def set_patient(self, patient_number):
        self.patient_number = patient_number

    def set_source(self, data_path):
        self.data_directory = data_path
