class Callbacks:
  """
  Handles all registered callbacks for Hooks

  """
  # Define the available callbacks
  _callbacks= {

    'on_pretrain_routine_start': [],
    'on_pretrain_routine_end':[],

    'on_train_start': [],
    'on_train_epoch_start': [],
    'on_train_batch_start': [],
    'optimizer_step': [],
    'on_before_zero_grad': [],
    'on_train_batch_end': [],
    'on_train_epoch_end': [],

    'on_val_start': [],
    'on_val_batch_start': [],
    'on_val_image_end': [],
    'on_val_batch_end': [],
    'on_val_end': [],

    'on_fit_epoch_end': [],
    'on_model_save': [],
    'on_train_end': []

    'teardown': []

  }
  
  def register_action(self, hook, name= '', callback = None):
    """
    Register a new action to a callback hook

    arguments:
      hook -- The callback hook name to register the action to.
      name -- The name of the action for the later reference.
      callback -- The callback to fire


    """
    assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
    assert callable(callback), f"callback '{callback}' is not callable"
    self._callbacks[hook].append({'name': name, 'callback': callback})
  
  def get_registered_actions(self, hook= None):

    """
    Returns all the registered actions by the callback hook

    arguments:
      hook -- The name of the hook to check, default to all.
    
    """
    if hook:
      return self._callbacks[hook]
    else:
      return self._callbacks

  def run(self, hook, *args, **kwargs):

    """

    Loop through the registered action and fire all the callbacks

    arguments:
      hook -- The name of the hook to check, defaults to all. 
      args -- Arguments to recieve from.
      kwargs -- keyword arguments ro recieve from. 

    """

    assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
    for logger in self._callbacks[hook]:
      logger['callback'](*args, **kwargs)

