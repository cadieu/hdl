def test_display_multilayer():

    from hdl.display import display_multilayer

    # load a multilayer model
    from hdl.models import ConvSparseSlowModel, SparseSlowModel
    from hdl.hierarchical_learners import HDL
    from hdl.config import state_dir
    import os

    from machines_vs_neurons.machines import MODEL_YouTubeFacesCrop60

    load_models = [
        os.path.split(MODEL_YouTubeFacesCrop60)[0],
        'M_vs_N_HDL_2layer_faces_2013-03-07_22-12-09/layer_1_2013-03-07_22-12-09',
    ]
    output_functions = ['proj_rect_sat', 'proj']

    total_layers = len(load_models)
    model_sequence = []
    for layer in range(total_layers):

        m = SparseSlowModel()

        m.load(os.path.join(state_dir, load_models[layer], 'model.model'))

        model_sequence.append(m)

    hdl_learner = HDL(model_sequence=model_sequence,
                      datasource=None,
                      output_functions=output_functions,)

    display_multilayer(hdl_learner)

if __name__ == '__main__':
    test_display_multilayer()