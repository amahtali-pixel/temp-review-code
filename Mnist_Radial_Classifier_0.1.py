import torch
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import os
import pickle
import time
import json
from collections import defaultdict
from sklearn.datasets import fetch_openml
from scipy import ndimage

"""
UNIFIED DIGIT RECOGNITION SYSTEM WITH DUAL PATTERN EXTRACTION AND DIGIT RECENTERING
- Double Rotary Patterns (8 values) + Simple Rotary Patterns (24 values)
- Consistent Zone Identification (16 zones, 4x4 grid)
- Three-Stage Validation System
- Digit Recentering (Center of Gravity Adjustment)                  
- Wrong Prediction Display
- CONFIDENCE BASED ON RAW VOTE MARGIN (first_best - second_best)/(total_votes + 1)
- Confidence statistics every 100 images
- DIGIT 1 SPECIAL HANDLING: When tied with any class, predict digit 1
"""

# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Color codes for better output visualization
YELLOW = '\033[93m'
PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
ENDC = '\033[0m'


class DigitRecenterer:
    """Handles digit recentering using center of gravity calculation"""

    def __init__(self, target_size=28, padding=2):
        self.target_size = target_size
        self.padding = padding
        self.final_size = target_size

    def calculate_center_of_gravity(self, image):
        """Calculate the center of gravity of the digit"""
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image.copy()

        # Ensure image is 2D
        if image_np.ndim > 2:
            image_np = image_np.squeeze()

        # Create binary mask
        binary_image = image_np > 0.1

        # Calculate center of gravity
        y_coords, x_coords = np.where(binary_image)

        if len(y_coords) == 0 or len(x_coords) == 0:
            # No visible pixels, return center
            return self.target_size // 2, self.target_size // 2

        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)

        return center_y, center_x

    def calculate_required_shift(self, center_y, center_x):
        """Calculate how much to shift the image to center it"""
        target_center = self.target_size // 2
        shift_y = target_center - center_y
        shift_x = target_center - center_x

        # Limit shifts to avoid moving too far
        max_shift = self.target_size // 4
        shift_y = np.clip(shift_y, -max_shift, max_shift)
        shift_x = np.clip(shift_x, -max_shift, max_shift)

        return int(round(shift_y)), int(round(shift_x))

    def apply_shift(self, image, shift_y, shift_x):
        """Apply shift to recenter the digit"""
        if isinstance(image, torch.Tensor):
            return self._shift_tensor(image, shift_y, shift_x)
        else:
            return self._shift_numpy(image, shift_y, shift_x)

    def _shift_tensor(self, image, shift_y, shift_x):
        """Shift tensor image using roll operation (simpler and faster)"""
        if image.dim() == 2:
            # 2D tensor
            shifted = torch.roll(image, shifts=(shift_y, shift_x), dims=(0, 1))
            # Fill edges with zeros
            if shift_y > 0:
                shifted[:shift_y, :] = 0
            elif shift_y < 0:
                shifted[shift_y:, :] = 0
            if shift_x > 0:
                shifted[:, :shift_x] = 0
            elif shift_x < 0:
                shifted[:, shift_x:] = 0
            return shifted
        else:
            # Handle other dimensions if needed
            return image

    def _shift_numpy(self, image, shift_y, shift_x):
        """Shift numpy image using roll operation"""
        if image.ndim > 2:
            image = image.squeeze()

        shifted = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))

        # Fill edges with zeros
        if shift_y > 0:
            shifted[:shift_y, :] = 0
        elif shift_y < 0:
            shifted[shift_y:, :] = 0
        if shift_x > 0:
            shifted[:, :shift_x] = 0
        elif shift_x < 0:
            shifted[:, shift_x:] = 0

        return shifted

    def recenter_digit(self, image):
        """Main method to recenter a digit based on center of gravity"""
        # Calculate center of gravity
        center_y, center_x = self.calculate_center_of_gravity(image)

        # Calculate required shift
        shift_y, shift_x = self.calculate_required_shift(center_y, center_x)

        # Apply shift if needed
        if abs(shift_y) > 1 or abs(shift_x) > 1:
            recentered = self.apply_shift(image, shift_y, shift_x)
            print(
                f"{CYAN}Recentered: shift=({shift_y}, {shift_x}), original_center=({center_y:.1f}, {center_x:.1f}){ENDC}")
            return recentered
        else:
            print(f"{CYAN}Already centered: center=({center_y:.1f}, {center_x:.1f}){ENDC}")
            return image


class ConfidenceMonitor:
    """Monitors and displays confidence statistics"""

    def __init__(self):
        self.confidences = []
        self.stages = []
        self.correct_confidences = []
        self.wrong_confidences = []

    def add_confidence(self, confidence, stage, is_correct):
        """Add confidence data"""
        self.confidences.append(confidence)
        self.stages.append(stage)
        if is_correct:
            self.correct_confidences.append(confidence)
        else:
            self.wrong_confidences.append(confidence)

    def display_statistics(self, image_count):
        """Display confidence statistics"""
        if len(self.confidences) == 0:
            return

        print(f"\n{CYAN}=== CONFIDENCE STATISTICS (Images {image_count}) ==={ENDC}")

        # Overall statistics
        print(f"{BLUE}Overall Confidence: mean={np.mean(self.confidences):.4f}, "
              f"median={np.median(self.confidences):.4f}, "
              f"std={np.std(self.confidences):.4f}{ENDC}")

        # Correct vs Wrong
        if len(self.correct_confidences) > 0:
            print(f"{GREEN}Correct predictions: mean={np.mean(self.correct_confidences):.4f}, "
                  f"median={np.median(self.correct_confidences):.4f}, "
                  f"n={len(self.correct_confidences)}{ENDC}")

        if len(self.wrong_confidences) > 0:
            print(f"{RED}Wrong predictions: mean={np.mean(self.wrong_confidences):.4f}, "
                  f"median={np.median(self.wrong_confidences):.4f}, "
                  f"n={len(self.wrong_confidences)}{ENDC}")

        # Stage distribution
        stage_counts = {}
        for stage in self.stages:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        print(f"{YELLOW}Stage distribution:{ENDC}")
        for stage, count in stage_counts.items():
            percentage = (count / len(self.stages)) * 100
            print(f"  {stage}: {count} ({percentage:.1f}%)")

        # Confidence distribution
        bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        hist, _ = np.histogram(self.confidences, bins=bins)
        print(f"{YELLOW}Confidence distribution:{ENDC}")
        for i in range(len(bins) - 1):
            percentage = (hist[i] / len(self.confidences)) * 100
            print(f"  {bins[i]:.3f}-{bins[i + 1]:.3f}: {hist[i]} ({percentage:.1f}%)")


class WrongPredictionLogger:
    """Logs and displays wrong predictions with detailed information"""

    def __init__(self, max_display=10):
        self.wrong_predictions = []
        self.max_display = max_display
        self.wrong_count = 0

    def add_wrong_prediction(self, image_idx, true_label, predicted_class,
                             confidence, match_counts, image_data, stage_used):
        """Add a wrong prediction to the log"""
        wrong_pred = {
            'image_idx': image_idx,
            'true_label': true_label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'match_counts': match_counts.copy(),
            'image_data': image_data.cpu().numpy() if torch.is_tensor(image_data) else image_data,
            'stage_used': stage_used
        }
        self.wrong_predictions.append(wrong_pred)
        self.wrong_count += 1

    def display_wrong_prediction(self, wrong_pred):
        """Display detailed information about a wrong prediction"""
        print(f"\n{RED}🚨 WRONG PREDICTION 🚨{ENDC}")
        print(
            f"{RED}Image {wrong_pred['image_idx']}: True={wrong_pred['true_label']}, Predicted={wrong_pred['predicted_class']}{ENDC}")
        print(f"{RED}Confidence: {wrong_pred['confidence']:.4f} (Stage: {wrong_pred['stage_used']}){ENDC}")

        # Show match counts for all classes
        print(f"{YELLOW}Match counts:{ENDC}")
        for digit in range(10):
            count = wrong_pred['match_counts'].get(digit, 0)
            marker = " ← WRONG" if digit == wrong_pred['predicted_class'] else ""
            marker = " ← TRUE" if digit == wrong_pred['true_label'] else marker
            print(f"  {digit}: {count}{marker}")

        # Show the image as ASCII art
        self.display_digit_ascii(wrong_pred['image_data'])

    def display_digit_ascii(self, image_data):
        """Display digit as ASCII art for wrong predictions"""
        if image_data.ndim > 2:
            image_data = image_data.squeeze()

        # Normalize and resize for ASCII display
        img_normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
        img_resized = img_normalized[::2, ::2]  # Downsample for ASCII

        print(f"{CYAN}Digit preview:{ENDC}")
        ascii_chars = " .:-=+*#%@"

        for i in range(img_resized.shape[0]):
            line = ""
            for j in range(img_resized.shape[1]):
                intensity = img_resized[i, j]
                char_idx = min(int(intensity * len(ascii_chars)), len(ascii_chars) - 1)
                line += ascii_chars[char_idx] * 2  # Double width for better aspect ratio
            print(f"{CYAN}{line}{ENDC}")

    def display_summary(self, total_images):
        """Display summary of wrong predictions"""
        if self.wrong_count > 0:
            accuracy = (total_images - self.wrong_count) / total_images * 100
            print(f"\n{RED}=== WRONG PREDICTIONS SUMMARY ==={ENDC}")
            print(
                f"{RED}Total wrong: {self.wrong_count}/{total_images} ({self.wrong_count / total_images * 100:.1f}%){ENDC}")
            print(f"{GREEN}Accuracy: {accuracy:.1f}%{ENDC}")

            # Show wrong prediction distribution
            wrong_by_true = defaultdict(int)
            wrong_by_predicted = defaultdict(int)
            wrong_by_stage = defaultdict(int)

            for wp in self.wrong_predictions:
                wrong_by_true[wp['true_label']] += 1
                wrong_by_predicted[wp['predicted_class']] += 1
                wrong_by_stage[wp['stage_used']] += 1

            print(f"{YELLOW}Wrong predictions by true label:{ENDC}")
            for digit in sorted(wrong_by_true.keys()):
                print(f"  {digit}: {wrong_by_true[digit]}")

            print(f"{YELLOW}Wrong predictions by predicted label:{ENDC}")
            for digit in sorted(wrong_by_predicted.keys()):
                print(f"  {digit}: {wrong_by_predicted[digit]}")

            print(f"{YELLOW}Wrong predictions by stage:{ENDC}")
            for stage in sorted(wrong_by_stage.keys()):
                print(f"  {stage}: {wrong_by_stage[stage]}")

            # Display first few wrong predictions in detail
            print(
                f"\n{YELLOW}Detailed view of first {min(self.max_display, len(self.wrong_predictions))} wrong predictions:{ENDC}")
            for i, wrong_pred in enumerate(self.wrong_predictions[:self.max_display]):
                self.display_wrong_prediction(wrong_pred)
                if i < min(self.max_display, len(self.wrong_predictions)) - 1:
                    print(f"{YELLOW}{'=' * 50}{ENDC}")


class ThreeStageValidator:
    def __init__(self):
        # Load pixel density statistics
        self.pixel_stats = self.load_pixel_statistics()
        self.simple_rotary_patterns = self.load_simple_rotary_patterns()

    def load_pixel_statistics(self):
        """Load pixel density statistics"""
        try:
            with open('pixel_density_statistics.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"{RED}Warning: pixel_density_statistics.json not found{ENDC}")
            return None

    def load_simple_rotary_patterns(self):
        """Load simple rotary patterns (24 values) for validation"""
        simple_patterns = {}
        for digit in range(10):
            try:
                with open(f'binary_{digit}_train_with_zones.pkl', 'rb') as f:
                    data = pickle.load(f)
                    simple_patterns[digit] = data['patterns_with_zones']
            except FileNotFoundError:
                print(f"{RED}Warning: binary_{digit}_train_with_zones.pkl not found{ENDC}")
                simple_patterns[digit] = []
        return simple_patterns

    def count_active_pixels(self, image):
        """Count non-zero pixels in an image"""
        return torch.count_nonzero(image).item()

    def get_zone_id(self, y, x, image_size=28, grid_size=4):
        """Consistent zone calculation for both pattern types"""
        zone_size = image_size / grid_size
        zone_y = int(y / zone_size)
        zone_x = int(x / zone_size)
        return zone_y * grid_size + zone_x

    def stage2_simple_rotary(self, image_simple_patterns, tied_classes):
        """Stage 2: Simple rotary pattern matching"""
        if not self.simple_rotary_patterns:
            return tied_classes

        simple_matches = {}
        for class_id in tied_classes:
            if class_id in self.simple_rotary_patterns:
                ref_patterns = self.simple_rotary_patterns[class_id]
                matches = 0
                for img_pattern in image_simple_patterns:
                    img_zone = self.get_zone_id(img_pattern['position'][0], img_pattern['position'][1])
                    # Find matching patterns in same zone
                    for ref_pattern in ref_patterns:
                        if ref_pattern['zone'] == img_zone:
                            distance = np.linalg.norm(
                                np.array(ref_pattern['pattern']) - np.array(img_pattern['pattern']))
                            if distance < 0.2:  # Slightly higher tolerance for binary patterns
                                matches += 1
                simple_matches[class_id] = matches
            else:
                simple_matches[class_id] = 0

        if not simple_matches:
            return tied_classes

        max_matches = max(simple_matches.values())
        return [cls for cls, count in simple_matches.items() if count == max_matches]

    def stage3_pixel_density(self, tied_classes, image_pixel_count):
        """Stage 3: Pixel density validation"""
        if self.pixel_stats is None:
            return tied_classes[0] if tied_classes else None

        best_match = None
        smallest_diff = float('inf')

        for class_id in tied_classes:
            class_str = str(class_id)
            if class_str in self.pixel_stats['density_profiles']:
                stats = self.pixel_stats['density_profiles'][class_str]
                avg_pixels = stats['avg_pixels']
                diff = abs(avg_pixels - image_pixel_count)

                if diff < smallest_diff:
                    smallest_diff = diff
                    best_match = class_id

        return best_match


class DualPatternMatcher:
    def __init__(self, device='cuda', tolerance=0.19, batch_size=32, enable_recentering=True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.enable_recentering = enable_recentering
        self.ref_arrays = {}
        self.ref_zones = {}
        self.stats = {'total_processed': 0, 'correct_predictions': 0}
        self.validator = ThreeStageValidator()
        self.recenterer = DigitRecenterer()
        self.wrong_logger = WrongPredictionLogger(max_display=5)
        self.confidence_monitor = ConfidenceMonitor()

        # Pattern collections
        self.correct_patterns = []
        self.incorrect_patterns = []

        # Precompute constants for double rotary
        self.shifts = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1],
                                    [0, 1], [1, -1], [1, 0], [1, 1]],
                                   dtype=torch.long)

        print(f"{GREEN}DualPatternMatcher initialized on device: {self.device}{ENDC}")
        print(f"{GREEN}Digit recentering: {'ENABLED' if enable_recentering else 'DISABLED'}{ENDC}")
        print(f"{GREEN}Using confidence formula: (first_best - second_best)/(total_votes + 1){ENDC}")
        print(f"{GREEN}DIGIT 1 RULE: When tied with any class, predict digit 1{ENDC}")

    def preprocess_image(self, image):
        """Preprocess image with optional recentering"""
        if self.enable_recentering:
            return self.recenterer.recenter_digit(image)
        return image

    def get_zone_id_batch(self, coords, image_size=28, grid_size=4):
        """Consistent zone calculation for batch processing"""
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()

        y, x = coords[:, 0], coords[:, 1]
        zone_size = image_size / grid_size

        zone_y = (y / zone_size).astype(int)
        zone_x = (x / zone_size).astype(int)

        return zone_y * grid_size + zone_x

    def fast_edge_detection(self, batch):
        """Detect edges in images using optimized convolution"""
        batch = batch.to(self.device)

        if batch.dim() == 3:
            batch = batch.unsqueeze(1)

        padded = F.pad(batch, (1, 1, 1, 1), mode='constant', value=0)

        # Create and configure kernel
        kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]],
                              dtype=torch.float32, device=self.device)
        kernel = kernel.repeat(batch.size(1), 1, 1, 1)

        neighbor_counts = F.conv2d(padded, kernel)
        edges = ((batch > 0) & (neighbor_counts < 8) & (neighbor_counts > 0))
        return edges.squeeze(1) if edges.size(1) == 1 else edges

    def extract_double_rotary_patterns(self, images_batch):
        """Extract double rotary patterns (8 values)"""
        batch_size = images_batch.size(0)
        images_batch = images_batch.to(self.device)

        edges = self.fast_edge_detection(images_batch)
        all_features = []

        for i in range(batch_size):
            image_edges = edges[i]
            edge_mask = image_edges > 0

            if not torch.any(edge_mask):
                continue

            edge_coords = torch.stack(torch.where(edge_mask), dim=1)
            padded_img = F.pad(images_batch[i].unsqueeze(0), (1, 1, 1, 1),
                               mode='constant', value=0).squeeze(0)

            # Extract neighbor values
            y_coords = edge_coords[:, 0].unsqueeze(1) + self.shifts[:, 0].to(self.device) + 1
            x_coords = edge_coords[:, 1].unsqueeze(1) + self.shifts[:, 1].to(self.device) + 1

            y_coords = torch.clamp(y_coords, 0, padded_img.size(0) - 1)
            x_coords = torch.clamp(x_coords, 0, padded_img.size(1) - 1)

            neighbor_values = padded_img[y_coords, x_coords]
            rotary_patterns = neighbor_values + torch.roll(neighbor_values, shifts=1, dims=1)

            coordinates = edge_coords[:, :2].float()
            batch_indices = torch.full((len(edge_coords), 1), i,
                                       dtype=torch.float32, device=self.device)

            features = torch.cat([rotary_patterns, coordinates, batch_indices], dim=1)
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0).numpy() if all_features else np.array([])

    def extract_simple_rotary_patterns(self, image):
        """Extract simple rotary patterns (24 values) from single image"""
        image_np = image.cpu().numpy() if torch.is_tensor(image) else image
        binary_image = (image_np > 0.5).astype(int).reshape(28, 28)

        patterns = []

        # Find edge pixels
        for y in range(28):
            for x in range(28):
                if binary_image[y, x] > 0 and self.is_edge_pixel(binary_image, y, x):
                    pattern = self.get_clockwise_neighborhood(binary_image, y, x)
                    zone = self.validator.get_zone_id(y, x)
                    patterns.append({
                        'pattern': pattern,
                        'zone': zone,
                        'position': (y, x)
                    })

        return patterns

    def is_edge_pixel(self, digit, y, x):
        """Check if pixel is on the edge"""
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < 28 and 0 <= nx < 28:
                    if digit[ny, nx] == 0:
                        return True
                else:
                    return True
        return False

    def get_clockwise_neighborhood(self, binary_image, center_y, center_x):
        """Extract 24 neighborhood pixels in clockwise order"""
        neighborhood = []
        positions = [
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            (-1, 2), (0, 2), (1, 2), (2, 2),
            (2, 1), (2, 0), (2, -1), (2, -2),
            (1, -2), (0, -2), (-1, -2),
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1),
            (1, 0), (1, -1),
            (0, -1)
        ]

        for dy, dx in positions:
            ny, nx = center_y + dy, center_x + dx
            if 0 <= ny < 28 and 0 <= nx < 28:
                neighborhood.append(binary_image[ny, nx])
            else:
                neighborhood.append(0)

        return neighborhood

    def load_reference_patterns(self):
        """Load reference patterns for both methods"""
        self.ref_arrays = {}
        self.ref_zones = {}

        for class_digit in range(10):
            # Load double rotary patterns
            filename = f"clustered_01_{class_digit}.pkl"
            try:
                with open(filename, 'rb') as f:
                    patterns = pickle.load(f)

                if isinstance(patterns, np.ndarray) and patterns.size > 0:
                    self.ref_arrays[class_digit] = patterns
                    self.ref_zones[class_digit] = self.get_zone_id_batch(patterns[:, 8:10])
                elif isinstance(patterns, list) and len(patterns) > 0:
                    ref_array = np.array(patterns)
                    self.ref_arrays[class_digit] = ref_array
                    self.ref_zones[class_digit] = self.get_zone_id_batch(ref_array[:, 8:10])
                else:
                    self.ref_arrays[class_digit] = None
                    self.ref_zones[class_digit] = None

                count = len(patterns) if hasattr(patterns, '__len__') else 0
                print(f"{GREEN}Loaded {count} double rotary patterns for class {class_digit}{ENDC}")

            except FileNotFoundError:
                print(f"{RED}Warning: {filename} not found{ENDC}")
                self.ref_arrays[class_digit] = None
                self.ref_zones[class_digit] = None

    def count_pattern_matches(self, new_patterns):
        """Count pattern matches with reference classes"""
        match_counts = {i: 0 for i in range(10)}

        if len(new_patterns) == 0:
            return match_counts

        new_coords = new_patterns[:, 8:10]
        new_zones = self.get_zone_id_batch(new_coords)
        new_rotary = new_patterns[:, :8]
        unique_zones = np.unique(new_zones)

        for class_digit in range(10):
            if self.ref_arrays[class_digit] is None or len(self.ref_arrays[class_digit]) == 0:
                continue

            ref_rotary = self.ref_arrays[class_digit][:, :8]
            ref_zones_class = self.ref_zones[class_digit]

            for zone in unique_zones:
                new_in_zone = new_zones == zone
                if not np.any(new_in_zone):
                    continue

                ref_in_zone = ref_zones_class == zone
                if not np.any(ref_in_zone):
                    continue

                new_zone_patterns = new_rotary[new_in_zone]
                ref_zone_patterns = ref_rotary[ref_in_zone]

                # Vectorized distance calculation
                diff = new_zone_patterns[:, np.newaxis, :] - ref_zone_patterns[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=2)
                min_distances = np.min(distances, axis=1)

                match_counts[class_digit] += np.sum(min_distances <= self.tolerance)

        return match_counts

    def calculate_confidence_from_margin(self, match_counts):
        """
        Calculate confidence based on raw vote margin:
        confidence = (first_best - second_best) / (total_votes + 1)

        This formula ensures:
        - Margin = 0 (tie) → confidence = 0
        - Small margin → very low confidence
        - Larger margin → higher confidence
        """
        if not match_counts:
            return 0.0

        # Get top two scores
        sorted_counts = sorted(match_counts.values(), reverse=True)
        first = sorted_counts[0]
        second = sorted_counts[1] if len(sorted_counts) > 1 else 0

        # Calculate margin
        margin = first - second

        # Total votes
        total = sum(match_counts.values())

        if total == 0:
            return 0.0

        # Confidence formula
        confidence = margin / (total + 1)

        return confidence

    def three_stage_prediction(self, match_counts, simple_patterns, double_patterns, image_data, true_label):
        """Three-stage prediction with confidence based ONLY on raw vote margin"""
        stage_used = "Stage 1"

        # Calculate base confidence from raw vote margin
        confidence = self.calculate_confidence_from_margin(match_counts)

        # Stage 1: Double rotary pattern matching
        max_matches = max(match_counts.values())
        top_classes = [cls for cls, count in match_counts.items() if count == max_matches]

        # SPECIAL HANDLING: If digit 1 is in the tie, predict digit 1 immediately
        if 1 in top_classes and len(top_classes) > 1:
            print(f"{YELLOW}⚡ Stage 1 Tie involving digit 1: {top_classes}{ENDC}")
            print(f"{GREEN}✅ DIGIT 1 RULE APPLIED: Predicting 1{ENDC}")
            predicted_class = 1
            stage_used = "Stage 1 (Digit 1 Rule)"
            # Confidence already calculated from raw margin
            return predicted_class, confidence, stage_used

        if len(top_classes) == 1:
            # Stage 1 clear winner
            predicted_class = top_classes[0]
            return predicted_class, confidence, stage_used

        stage_used = "Stage 2"
        print(f"{YELLOW}⚡ Stage 1 Tie: {top_classes}{ENDC}")

        # Stage 2: Simple rotary pattern validation
        # Calculate simple pattern matches for tied classes
        simple_matches = {}
        for class_id in top_classes:
            if class_id in self.validator.simple_rotary_patterns:
                ref_patterns = self.validator.simple_rotary_patterns[class_id]
                matches = 0
                for img_pattern in simple_patterns:
                    img_zone = self.validator.get_zone_id(img_pattern['position'][0], img_pattern['position'][1])
                    # Find matching patterns in same zone
                    for ref_pattern in ref_patterns:
                        if ref_pattern['zone'] == img_zone:
                            distance = np.linalg.norm(
                                np.array(ref_pattern['pattern']) - np.array(img_pattern['pattern']))
                            if distance < 0.2:
                                matches += 1
                simple_matches[class_id] = matches
            else:
                simple_matches[class_id] = 0

        # Find Stage 2 winner
        if simple_matches:
            max_simple = max(simple_matches.values())
            stage2_classes = [cls for cls, count in simple_matches.items() if count == max_simple]
        else:
            stage2_classes = top_classes

        # SPECIAL HANDLING: If digit 1 is still in the tie after Stage 2, predict digit 1
        if 1 in stage2_classes and len(stage2_classes) > 1:
            print(f"{YELLOW}⚡ Stage 2 Tie involving digit 1: {stage2_classes}{ENDC}")
            print(f"{GREEN}✅ DIGIT 1 RULE APPLIED: Predicting 1{ENDC}")
            predicted_class = 1
            stage_used = "Stage 2 (Digit 1 Rule)"
            # Confidence remains from raw margin (which is 0 for a tie)
            return predicted_class, confidence, stage_used

        if len(stage2_classes) == 1:
            # Stage 2 resolved the tie
            predicted_class = stage2_classes[0]
            print(f"{GREEN}✅ Stage 2 resolved: {predicted_class}{ENDC}")
            # Confidence remains from raw margin (which is 0 for a tie)
            return predicted_class, confidence, stage_used

        stage_used = "Stage 3"
        print(f"{YELLOW}⚡ Stage 2 Tie: {stage2_classes}{ENDC}")

        # Stage 3: Pixel density validation
        image_pixels = self.validator.count_active_pixels(image_data)
        final_prediction = self.validator.stage3_pixel_density(stage2_classes, image_pixels)

        if final_prediction is not None:
            print(f"{GREEN}✅ Stage 3 resolved: {final_prediction} (Pixels: {image_pixels}){ENDC}")
            # Confidence remains from raw margin
            return final_prediction, confidence, stage_used

        # Fallback: Use first of top classes
        predicted_class = top_classes[0]
        stage_used = "Fallback"
        # Confidence remains from raw margin
        return predicted_class, confidence, stage_used

    def process_single_image(self, image, true_label, image_idx):
        """Process a single image with dual pattern validation and recentering"""
        print(f"\n{YELLOW}=== Processing Image {image_idx} (True class: {true_label}) ==={ENDC}")

        # Preprocess image with recentering
        original_image = image.clone()
        image = self.preprocess_image(image)

        # Extract both pattern types
        image_batch = image.unsqueeze(0)
        double_patterns = self.extract_double_rotary_patterns(image_batch)
        simple_patterns = self.extract_simple_rotary_patterns(image)

        if len(double_patterns) == 0 and len(simple_patterns) == 0:
            print(f"{YELLOW}No patterns found{ENDC}")
            result = {
                'image_idx': image_idx,
                'true_label': true_label,
                'predicted_class': None,
                'confidence': 0,
                'match_counts': {i: 0 for i in range(10)},
                'correct': False,
                'recentered': not torch.equal(original_image, image),
                'stage_used': 'No patterns'
            }
            return result

        print(f"{GREEN}Extracted {len(double_patterns)} double patterns, {len(simple_patterns)} simple patterns{ENDC}")

        # Count matches and analyze with three-stage validation
        match_counts = self.count_pattern_matches(double_patterns)
        predicted_class, confidence, stage_used = self.three_stage_prediction(
            match_counts, simple_patterns, double_patterns, image, true_label
        )

        # Check if prediction is correct
        is_correct = predicted_class == true_label if predicted_class is not None else False

        # Update statistics
        self.stats['total_processed'] += 1
        if is_correct:
            self.stats['correct_predictions'] += 1
            print(f"{GREEN}✅ Correct prediction! (Confidence: {confidence:.4f}, Stage: {stage_used}){ENDC}")
        else:
            print(f"{RED}❌ Wrong prediction! (Confidence: {confidence:.4f}, Stage: {stage_used}){ENDC}")
            # Log wrong prediction
            self.wrong_logger.add_wrong_prediction(
                image_idx, true_label, predicted_class, confidence,
                match_counts, image, stage_used
            )
            # Display wrong prediction immediately
            if len(self.wrong_logger.wrong_predictions) <= self.wrong_logger.max_display:
                self.wrong_logger.display_wrong_prediction(
                    self.wrong_logger.wrong_predictions[-1]
                )

        # Add to confidence monitor
        self.confidence_monitor.add_confidence(confidence, stage_used, is_correct)

        # Show instant accuracy
        if self.stats['total_processed'] > 0:
            accuracy = self.stats['correct_predictions'] / self.stats['total_processed']
            print(f"{BLUE}Instant Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%) - "
                  f"{self.stats['correct_predictions']}/{self.stats['total_processed']}{ENDC}")

        return {
            'image_idx': image_idx,
            'true_label': true_label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'match_counts': match_counts,
            'correct': is_correct,
            'double_patterns': len(double_patterns),
            'simple_patterns': len(simple_patterns),
            'recentered': not torch.equal(original_image, image),
            'stage_used': stage_used
        }

    def run(self, num_images=10000):
        """Main execution method"""
        print(f"{BLUE}Loading test images...{ENDC}")
        test_dataset = datasets.MNIST(root='./data', train=False, download=True)
        test_images = test_dataset.data[:num_images].float() / 255.0
        test_labels = test_dataset.targets[:num_images]

        print(f"{BLUE}Loading reference patterns...{ENDC}")
        self.load_reference_patterns()

        all_results = []
        start_time = time.time()

        # Process images
        for i in range(len(test_images)):
            result = self.process_single_image(test_images[i], test_labels[i].item(), i)
            all_results.append(result)

            # Display confidence statistics every 100 images
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"{BLUE}Processed {i + 1}/{len(test_images)} images "
                      f"({elapsed:.2f} seconds elapsed){ENDC}")
                # Display confidence statistics
                self.confidence_monitor.display_statistics(i + 1)

        # Final results
        self.print_final_results(all_results, start_time)

        # Final confidence statistics
        self.confidence_monitor.display_statistics(len(all_results))

        return all_results

    def print_final_results(self, all_results, start_time):
        """Print final accuracy and statistics"""
        elapsed = time.time() - start_time

        # Calculate recentering statistics
        recentered_count = sum(1 for result in all_results if result.get('recentered', False))
        recentered_percentage = (recentered_count / len(all_results)) * 100 if all_results else 0

        print(f"\n{PINK}=== PROCESSING COMPLETED ==={ENDC}")
        print(f"{PINK}Time: {elapsed:.2f} seconds{ENDC}")
        print(f"{PINK}Images processed: {self.stats['total_processed']}{ENDC}")
        print(f"{PINK}Images recentered: {recentered_count}/{len(all_results)} ({recentered_percentage:.1f}%){ENDC}")

        if self.stats['total_processed'] > 0:
            accuracy = self.stats['correct_predictions'] / self.stats['total_processed']
            print(f"{PINK}Final Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%){ENDC}")
            print(f"{PINK}Correct predictions: {self.stats['correct_predictions']}/"
                  f"{self.stats['total_processed']}{ENDC}")

        # Display wrong predictions summary
        self.wrong_logger.display_summary(len(all_results))

        # Save results
        with open('dual_pattern_results_with_unified_confidence.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        print(f"{GREEN}Results saved to dual_pattern_results_with_unified_confidence.pkl{ENDC}")


def main():
    """Main function"""
    # Initialize dual pattern matcher with recentering enabled
    matcher = DualPatternMatcher(
        device='cuda',
        tolerance=0.19,
        batch_size=32,
        enable_recentering=True
    )

    # Run the pattern matching
    results = matcher.run(num_images=10000)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()