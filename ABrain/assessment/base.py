from typing import Iterable, Tuple
from torchio.data import Subject
import torchio as tio
from torch import Tensor


class Supervisor(object):
    '''The supervisor who receiving the results of the `Inspector`.'''

    def __init__(self) -> None:
        pass

    def receive(self, subject: Subject, why: str):
        sid = subject['id']
        print(sid, why)


class Inspector(object):
    '''The inspector who submits the results to the `Supervisor`'''

    def __init__(self, supervisor: Supervisor = None) -> None:
        self.supervisor = supervisor

    def check(self, subject: Subject) -> Tuple[bool, str]:
        class_name = self.__class__.__name__
        why = "%s must implement `self.check(subject)`!" % (class_name)
        raise NotImplementedError(why)

    def __call__(self, subject: Subject) -> bool:
        is_ok, why = self.check(subject)
        if not is_ok and self.supervisor:
            self.supervisor.receive(subject, why)
        return is_ok


class CheckSomeOf(Inspector):
    '''Inspection of conformity.

    The inspectors check whether the object is qualified. When a inspector finds a nonconformity, the inspection of the object is terminated.
    '''

    def __init__(self, inspectors: Iterable[Inspector], supervisor: Supervisor = None) -> None:
        super().__init__(supervisor)
        self.inspectors = inspectors
        if supervisor is not None:
            for inspector in self.inspectors:
                inspector.supervisor = supervisor

    def __call__(self, subject: Subject) -> bool:
        for inspector in self.inspectors:
            is_ok = inspector(subject)
            if not is_ok:
                return False
        return True


class CheckAllOf(Inspector):
    '''Find all the unqualified problems.

    All the inspectors check in turn. It will not stop until all the inspectors have completed.
    '''

    def __init__(self, inspectors: Iterable[Inspector], supervisor: Supervisor = None) -> None:
        super().__init__(supervisor)
        self.inspectors = inspectors

    def __call__(self, subject: Subject) -> bool:
        is_ok = True
        for inspector in self.inspectors:
            flag = inspector(subject)
            is_ok = is_ok and flag
        return is_ok


class ShapeConsistency(Inspector):
    '''Check the consistency(shape and others).'''

    def __init__(self, supervisor: Supervisor = None) -> None:
        super().__init__(supervisor)

    def check(self, subject: Subject) -> Tuple[bool, str]:
        try:
            subject.check_consistent_attribute('shape')
        except RuntimeError as e:
            reason = "In the same observation object, different `shape` are recognized."
            return False, reason
        return True, ""


class LabelLegality(Inspector):
    '''Check that the label is valid.'''

    def __init__(self, labels: Iterable, supervisor: Supervisor = None) -> None:
        super().__init__(supervisor)
        self.labels = set(labels)

    def check(self, subject: Subject) -> Tuple[bool, str]:
        data: Tensor = subject['seg'][tio.DATA]
        labels = data.unique()
        label_illegal = []
        for each in labels:
            label = each.item()
            if label not in self.labels:
                label_illegal.append(label)
        if len(label_illegal) != 0:
            reason = "Illegal labels: " + str(label_illegal)
            return False, reason
        return True, ""
