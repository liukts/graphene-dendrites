import torch
import torch.jit
from norse.torch import LIFParameters
from norse.torch.functional.lif import LIFParametersJIT
from norse.torch.module.snn import SNNCell
from typing import NamedTuple, Optional, Tuple
from norse.torch.functional.threshold import threshold

class DENDFeedForwardState(NamedTuple):
    """State of dendrite

    Parameters:
        v (torch.Tensor): membrane potential (dendrite state)
        i (torch.Tensor): synaptic input current
    """

    v: torch.Tensor
    i: torch.Tensor

class DENDCell(SNNCell):
    """Module that computes a single euler-integration step of a
    leaky integrate-and-fire (LIF) neuron-model *without* recurrence and *without* time.

    More specifically it implements one integration step
    of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}}
        \\end{align*}

    Example:
        >>> data = torch.zeros(5, 2) # 5 batches, 2 neurons
        >>> l = LIFCell(2, 4)
        >>> l(data) # Returns tuple of (Tensor(5, 4), LIFState)

    Arguments:
        p (LIFParameters): Parameters of the LIF neuron model.
        sparse (bool): Whether to apply sparse activation functions (True) or not (False). Defaults to False.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(self, p: LIFParameters = LIFParameters(), **kwargs):
        super().__init__(
            activation=dend_feed_forward_step,
            activation_sparse= dend_feed_forward_step_sparse,
            state_fallback=self.initial_state,
            p=LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> DENDFeedForwardState:
        state = DENDFeedForwardState(
            v=torch.full(
                input_tensor.shape,
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state

@torch.jit.script
def _dend_feed_forward_step_jit(
    input_tensor: torch.Tensor,
    state: DENDFeedForwardState,
    p: LIFParametersJIT,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DENDFeedForwardState]:  # pragma: no cover

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    v_decayed = state.v + dv
    # v_decayed = torch.where(v_decayed > 0, v_decayed, torch.tensor(0,dtype=torch.float32,
    #                                               device=torch.device("cuda")))

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute current jumps
    i_new = i_decayed + input_tensor

    i_new = i_new + torch.empty_like(i_new).normal_(mean=torch.tensor(0.0),std=torch.tensor(0.005)) * dt

    return v_decayed, DENDFeedForwardState(v=v_decayed, i=i_new)

def dend_feed_forward_step(
    input_tensor: torch.Tensor,
    state: Optional[DENDFeedForwardState],
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DENDFeedForwardState]:
    r"""Computes a single euler-integration step for a lif dendrite-model.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration
    step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}}
        \end{align*}

    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        state (LIFFeedForwardState): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    # if norse.utils.IS_OPS_LOADED:
    #     try:
    #         z, v, i = norse_op.lif_super_feed_forward_step(input_tensor, state, p, dt)
    #         return z, GFETFeedForwardState(v=v, i=i)
    #     except NameError:  # pragma: no cover
    #         pass
    jit_params = LIFParametersJIT(
        tau_syn_inv=p.tau_syn_inv,
        tau_mem_inv=p.tau_mem_inv,
        v_leak=p.v_leak,
        v_th=p.v_th,
        v_reset=p.v_reset,
        method=p.method,
        alpha=torch.as_tensor(p.alpha),
    )
    return _dend_feed_forward_step_jit(input_tensor, state=state, p=jit_params, dt=dt)

def dend_feed_forward_step_sparse(
    input_tensor: torch.Tensor,
    state: DENDFeedForwardState,
    p: LIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DENDFeedForwardState]:  # pragma: no cover
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute current jumps
    i_new = i_decayed + input_tensor
    
    i_new = i_new + torch.empty_like(i_new).normal_(mean=torch.tensor(0.0),std=torch.tensor(0.005)) * dt

    return v_decayed.to_sparse(), DENDFeedForwardState(v=v_decayed, i=i_new)