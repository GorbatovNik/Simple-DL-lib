from numpy import ndarray

from utils.np_utils import assert_same_shape


class Operation(object):

    def __init__(self):
        pass

    def forward(self,
                input_: ndarray,
                inference: bool=False) -> ndarray:

        self.input_ = input_

        self.output = self._output(inference)

        return self.output

    def backward(self, output_grad: ndarray, skeep_param_grads = None) -> ndarray:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self, inference: bool) -> ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()


class ParamOperation(Operation):

    def __init__(self, param: ndarray) -> ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray, skeep_param_grads = False) -> ndarray:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        if not skeep_param_grads:
            self.param_grad = self._param_grad(output_grad)
        else:
            self.param_grad = None

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()
