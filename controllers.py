class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3


class PIDController(BaseController):
  def __init__(self, k_p=0.044, k_i=0.1, k_d=-0.035):
    self.k_p = k_p
    self.k_i = k_i
    self.k_d = k_d
    self.integral = 0.0
    self.prev_error = 0.0

  def update(self, target_lataccel, current_lataccel, state):
    error = target_lataccel - current_lataccel
    self.integral += error
    derivative = error - self.prev_error

    steer_action = (
      self.k_p * error
      + self.k_i * self.integral
      + self.k_d * derivative
    )

    self.prev_error = error
    return steer_action


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
}
