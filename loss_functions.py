import torch
from torch import nn


class EulerAnglesLoss(nn.Module):
    def __init__(self):
        """
        Custom loss function for Euler angles that handles periodicity and works with angles in degrees
        """
        super(EulerAnglesLoss, self).__init__()

    def normalize_angle(self, angle):
        """
        Normalize angle to [-180, 180] range
        Args:
            angle (torch.Tensor): Input angle in degrees
        Returns:
            torch.Tensor: Normalized angle in [-180, 180] range
        """
        return ((angle + 180) % 360) - 180

    def forward(self, pred, target):
        """
        Calculate the loss between predicted and target Euler angles
        Args:
            pred (torch.Tensor): Predicted Euler angles [batch_size, 3] in degrees
            target (torch.Tensor): Target Euler angles [batch_size, 3] in degrees
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Normalize both predicted and target angles to [-180, 180] range
        pred_normalized = torch.stack([self.normalize_angle(a) for a in pred.t()]).t()
        target_normalized = torch.stack([self.normalize_angle(a) for a in target.t()]).t()

        # Calculate the angular difference considering periodicity
        diff = pred_normalized - target_normalized
        diff_normalized = torch.stack([self.normalize_angle(d) for d in diff.t()]).t()

        # Calculate MSE loss on the normalized differences
        loss = torch.mean(diff_normalized ** 2)

        return loss




class QuaternionNormLoss(nn.Module):
    def __init__(self, norm_weight=0.1):
        super(QuaternionNormLoss, self).__init__()
        self.norm_weight = norm_weight

    def forward(self, pred, target):
        # Orientation loss
        inner_product = torch.sum(pred * target, dim=-1)
        orientation_loss = 1 - torch.abs(inner_product)

        # Unit norm constraint
        norm_loss = torch.abs(torch.sum(pred * pred, dim=-1) - 1.0)

        return torch.mean(orientation_loss + self.norm_weight * norm_loss)


class AngularMSELoss(nn.Module):
    def __init__(self, period=360.0):
        """
        Custom loss function for angular values that handles wrapping.

        Args:
            period (float): The period of the angle in degrees (360 for full circle, 180 for half circle)
        """
        super(AngularMSELoss, self).__init__()
        self.period = period

    def forward(self, pred, target):
        """
        Calculate the MSE loss accounting for angular wrapping.

        Args:
            pred (torch.Tensor): Predicted angles in degrees (batch_size x 3 for roll, pitch, yaw)
            target (torch.Tensor): Target angles in degrees (batch_size x 3 for roll, pitch, yaw)

        Returns:
            torch.Tensor: Mean squared angular difference
        """
        # Calculate the absolute difference
        diff = pred - target

        # Handle wrapping for each angle separately
        wrapped_diff = torch.remainder(diff + self.period / 2, self.period) - self.period / 2

        # Calculate MSE on the wrapped differences
        return torch.mean(wrapped_diff ** 2)