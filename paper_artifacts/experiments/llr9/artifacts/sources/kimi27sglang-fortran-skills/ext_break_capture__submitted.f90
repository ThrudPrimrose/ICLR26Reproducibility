subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(inout) :: out_index
  real(c_double), intent(inout) :: out_value

  integer(c_int64_t), parameter :: PAR_THRESHOLD = 10000_c_int64_t

  out_index = -1_c_int64_t
  out_value = -1.0_c_double

  if (LEN_1D < PAR_THRESHOLD) then
    call serial_find_first(a, LEN_1D, out_index, out_value)
  else
    call parallel_find_first(a, LEN_1D, out_index, out_value)
  end if

contains
  subroutine serial_find_first(a, n, out_index, out_value)
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: n
    real(c_double), intent(in) :: a(n)
    integer(c_int64_t), intent(inout) :: out_index
    real(c_double), intent(inout) :: out_value

    integer(c_int64_t) :: i
    real(c_double), parameter :: K = 1.0_c_double

    do i = 1, n
      if (a(i) > K) then
        out_index = i
        out_value = a(i)
        exit
      end if
    end do
  end subroutine serial_find_first

  subroutine parallel_find_first(a, n, out_index, out_value)
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: n
    real(c_double), intent(in) :: a(n)
    integer(c_int64_t), intent(inout) :: out_index
    real(c_double), intent(inout) :: out_value

    integer(c_int64_t) :: i, idx
    real(c_double), parameter :: K = 1.0_c_double

    idx = n + 1_c_int64_t

    !$omp parallel do simd reduction(min:idx) schedule(static)
    do i = 1, n
      if (a(i) > K) then
        idx = min(idx, i)
      end if
    end do
    !$omp end parallel do simd

    if (idx <= n) then
      out_index = idx
      out_value = a(idx)
    end if
  end subroutine parallel_find_first
end subroutine ext_break_capture_fp64
