import numpy as np
import scipy.interpolate

from numpy import ndarray
from abc import abstractmethod

class VectorFieldBase():
    """Base Class for concrete definitions to inherit."""
    @abstractmethod
    def get_vector(self, ps: ndarray, t:float) -> ndarray:
        """Return for each point p in ps a vector v."""
        pass


class StraightVectorField(VectorFieldBase):
    def __init__(self) -> None:
        self._direction = np.array([1.0, 0.0, 0.0])
    
    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, new_direction: ndarray):
        assert new_direction.shape == (3,), "Direction must be a (3,)-ndarray but is of shape %s" % str(new_direction.shape)
        self._direction = new_direction

    def get_vector(self, ps: ndarray, t:float) -> ndarray:
        return np.tile(self._direction, (ps.shape[0], 1))


class RotatingVectorField(VectorFieldBase):
    def __init__(self) -> None:
        self._center = np.array([.5, .5, 0])

    @property
    def center(self):
        return self._center
    
    @center.setter
    def center(self, new_center: ndarray):
        assert new_center.shape == (3,), "Center must be a (3,)-ndarray but is of shape %s" % str(new_center.shape)
        self._center = new_center

    def get_vector(self, ps:ndarray, t:float):
        xs = -(ps[:, 1] - self._center[1])
        ys =   ps[:, 0] - self._center[0]
        zs = np.zeros_like(xs)
        return np.vstack((xs, ys, zs)).T


class SourceVectorField(VectorFieldBase):
    _source_position = np.zeros(3, dtype=float)

    def get_vector(self, ps: ndarray, t:float) -> ndarray:
        return ps - self._source_position

    @property
    def source_position(self) -> ndarray:
        return self._source_position
    
    @source_position.setter
    def source_position(self, new_source_position: ndarray):
        assert new_source_position.shape == (3,), "Source position must be a (3,)-ndarray but is of shape %s" % str(new_source_position.shape)
        self._source_position = new_source_position


class SinkVectorField(VectorFieldBase):
    _sink_start_position = np.zeros(3, dtype=float)
    _sink_stop_position = np.zeros(3, dtype=float)
    _interp = scipy.interpolate.interp1d(_sink_start_position, _sink_stop_position)

    def get_vector(self, ps: ndarray, t:float) -> ndarray:
        assert 0 <= t <= 1
        return self.get_sink_position(t) - ps

    def get_sink_position(self, t) -> ndarray:
        return self._interp(t)
    
    @property
    def sink_start_position(self):
        return self._sink_start_position
    
    @sink_start_position.setter
    def sink_start_position(self, new_sink_position: ndarray):
        assert new_sink_position.shape == (3,), "Sink position must be a (3,)-ndarray but is of shape %s" % str(new_sink_position.shape)
        self._sink_start_position = new_sink_position
        self._update_interpolator()

    @property
    def sink_stop_position(self):
        return self._sink_stop_position
    
    @sink_stop_position.setter
    def sink_stop_position(self, new_sink_position: ndarray):
        assert new_sink_position.shape == (3,), "Sink position must be a (3,)-ndarray but is of shape %s" % str(new_sink_position.shape)
        self._sink_stop_position = new_sink_position
        self._update_interpolator()

    def _update_interpolator(self):
        self._interp = scipy.interpolate.interp1d(np.arange(2), np.array([self._sink_start_position, self._sink_stop_position]).T)


class DoubleOrbitVectorField(VectorFieldBase):
    _center_position_1 = np.zeros(3, dtype=float)
    _center_position_2 = np.array([0,10,0], dtype=float)
    _dist = 1.0

    def get_vector(self, ps: ndarray, t:float) -> ndarray:
        deltas1 = ps - self._center_position_1
        dists1 = np.linalg.norm(deltas1, axis=-1)[:,None] ** 2
        deltas2 = ps - self._center_position_2
        dists2 = np.linalg.norm(deltas2, axis=-1)[:,None] ** 2
        rot1 = deltas1 @ ((0,1,0), (-1,0,0), (0,0,1))
        rot2 = deltas2 @ ((0,-1,0), (1,0,0), (0,0,1))
        return rot1 / (dists1+.1) + rot2 / (dists2+.1)

    @property
    def center_position_1(self) -> ndarray:
        return self._center_position_1
    
    @property
    def center_position_2(self) -> ndarray:
        return self._center_position_2
    
    @property
    def dist(self) -> float:
        return self._dist
    
    @center_position_1.setter
    def center_position_1(self, new_center_position: ndarray):
        assert new_center_position.shape == (3,), "Center position must be a (3,)-ndarray but is of shape %s" % str(new_center_position.shape)
        self._center_position_1 = new_center_position

    @center_position_2.setter
    def center_position_2(self, new_center_position: ndarray):
        assert new_center_position.shape == (3,), "Center position must be a (3,)-ndarray but is of shape %s" % str(new_center_position.shape)
        self._center_position_2 = new_center_position

    @dist.setter
    def dist(self, new_dist:float):
        self._dist = new_dist


class WavesVectorField(VectorFieldBase):
    _offset = 0.0
    _magnitude = 1.0
    _frequency = 1
    _xspeed = 1
    _lcutoff = 0
    _rcutoff = 1

    def get_vector(self, ps: ndarray, t:float) -> ndarray:
        vecs = np.zeros_like(ps)
        period_len = self._frequency * 2 * np.pi
        xvals = self._xspeed
        yvals = np.sin((ps[:,0] + self._offset) * period_len) * self._magnitude
        vecs[:,0] = xvals
        vecs[:,1] = np.where( (self._lcutoff < ps[:,0]) & (ps[:,0] < self._rcutoff), yvals, 0)
        return vecs


class SpiralSinkVectorField(VectorFieldBase):
    _xshear = 1.0
    _sink_position = np.zeros(3, dtype=float)
    
    def get_vector(self, ps: ndarray, t:float) -> ndarray:
        mat = np.array([[-self._xshear, -1, 0], [self._xshear, -1, 0], [0, 0, 1]]).T
        return (ps + self._sink_position) @ mat

    @property
    def sink_position(self) -> ndarray:
        return self._sink_position
    
    @sink_position.setter
    def sink_position(self, new_sink_position: ndarray):
        assert new_sink_position.shape == (3,), "Sink position must be a (3,)-ndarray but is of shape %s" % str(new_sink_position.shape)
        self._sink_position = new_sink_position