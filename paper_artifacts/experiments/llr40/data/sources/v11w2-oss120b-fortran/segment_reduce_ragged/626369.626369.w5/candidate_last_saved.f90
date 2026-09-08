module segment_reduce_ragged_mod
  use iso_c_binding
contains
  subroutine segment_reduce_ragged_fp64(row_ptr, val, w, out, nseg) bind(C, name='segment_reduce_ragged_fp64')
    implicit none
    integer(c_int64_t), value :: nseg
    integer(c_int64_t), intent(in) :: row_ptr(*)
    real(c_double), intent(in) :: val(*)
    real(c_double), intent(in) :: w(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t) :: s, e
    integer(c_int64_t) :: start_idx, end_idx
    real(c_double) :: acc

    !$omp parallel do default(none) shared(row_ptr, val, w, out, nseg) private(s, start_idx, end_idx, e, acc)
    do s = 1, nseg
      start_idx = row_ptr(s) + 1_c_int64_t
      end_idx = row_ptr(s + 1)
      acc = 0.0_c_double
      do e = start_idx, end_idx
        acc = acc + val(e) * w(e)
      end do
      out(s) = acc
    end do
    !$omp end parallel do
  end subroutine segment_reduce_ragged_fp64
  subroutine segment_reduce_ragged_fp32(row_ptr, val, w, out, nseg) bind(C, name='segment_reduce_ragged_fp32')
    implicit none
    integer(c_int64_t), value :: nseg
    integer(c_int64_t), intent(in) :: row_ptr(*)
    real(c_float), intent(in) :: val(*)
    real(c_float), intent(in) :: w(*)
    real(c_float), intent(out) :: out(*)
    integer(c_int64_t) :: s, e
    integer(c_int64_t) :: start_idx, end_idx
    real(c_float) :: acc

    !$omp parallel do default(none) shared(row_ptr, val, w, out, nseg) private(s, start_idx, end_idx, e, acc)
    do s = 1, nseg
      start_idx = row_ptr(s) + 1_c_int64_t
      end_idx = row_ptr(s + 1)
      acc = 0.0_c_float
      do e = start_idx, end_idx
        acc = acc + val(e) * w(e)
      end do
      out(s) = acc
    end do
    !$omp end parallel do
  end subroutine segment_reduce_ragged_fp32

end module segment_reduce_ragged_mod
