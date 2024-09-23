import torch
from typing import Dict, List, Literal

class Config:
    DEVICE: torch.device = torch.device('cpu')
    COMPLEX_DTYPE: torch.dtype = torch.complex64
    REAL_DTYPE: torch.dtype = torch.float32
    BETA_DEFAULT: float = 2.5
    ALPHA_DEFAULT: float = 2.5
    ALPHA_DEFAULT_LIN: float = 1.8
    BETA_LUT: Dict[float, float] = {}
    ALPHA_LUT: Dict[float, float] = {}
    ALPHA_LUT_LIN: Dict[float, float] = {}
    BETA_PRUNE_DEFAULT = 1.5
    BETA_PRUNE = BETA_PRUNE_DEFAULT
    
    def set_beta(self, Q:float, beta: float):
        self.BETA_LUT[float(Q)] = beta        
        
    def set_alpha(self, Q: float, alpha: float, is_linear: bool = False):
        if is_linear: 
            self.ALPHA_LUT_LIN[float(Q)] = alpha
        else:
            self.ALPHA_LUT[float(Q)] = alpha
        
    def get_beta(self, Q: float) -> float:
        if float(Q) in self.BETA_LUT.keys(): return self.BETA_LUT[float(Q)]
        return self.BETA_DEFAULT
    
    def get_alpha(self, Q: float, is_linear: bool = False) -> float:
        if is_linear:            
            if float(Q) in self.ALPHA_LUT_LIN.keys(): return self.ALPHA_LUT_LIN[float(Q)]
            return self.ALPHA_DEFAULT_LIN
        else:
            if float(Q) in self.ALPHA_LUT.keys(): return self.ALPHA_LUT[float(Q)]
            return self.ALPHA_DEFAULT
        
    def set_beta_prune(self, beta_prune):
        self.BETA_PRUNE = beta_prune
        
    def get_beta_prune(self):
        return self.BETA_PRUNE
    
    def cuda(self):
        self.DEVICE = torch.device('cuda')
        
    def cpu(self):
        self.DEVICE = torch.device('cpu')
        
    def set_precision(self, prec: Literal['single', 'double']):
        assert(prec in ['single', 'double'])
        if prec == 'single':
            self.REAL_DTYPE = torch.float32
            self.COMPLEX_DTYPE = torch.complex64
        else:
            self.REAL_DTYPE = torch.float64
            self.COMPLEX_DTYPE = torch.complex128
            
cfg = Config()