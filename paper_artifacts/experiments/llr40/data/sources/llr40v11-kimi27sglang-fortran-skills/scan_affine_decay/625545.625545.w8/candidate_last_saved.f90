subroutine scan_affine_decay_fp64(y, c, x, n, workspace, workspace_size) bind(C, name="scan_affine_decay_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: y(n)
  real(c_double), intent(in) :: c(n), x(n)

  integer(c_int64_t), parameter :: serial_threshold = 512
  integer(c_int64_t) :: i, lo, hi, t, nt, use_threads
  real(c_double) :: p, prod, carry
  real(c_double), allocatable :: agg_A(:), agg_B(:), carry_in(:)

  if (n <= 0) return

  if (n <= serial_threshold) then
    y(1) = x(1)
    do i = 2, n
      y(i) = c(i) * y(i - 1) + x(i)
    end do
    return
  end if

  nt = omp_get_max_threads()
  use_threads = min(nt, n / serial_threshold)
  if (use_threads < 2) then
    y(1) = x(1)
    do i = 2, n
      y(i) = c(i) * y(i - 1) + x(i)
    end do
    return
  end if

  allocate(agg_A(0:use_threads - 1), agg_B(0:use_threads - 1), carry_in(0:use_threads - 1))

  !$omp parallel do schedule(static) default(shared) private(t, lo, hi, i, p, prod)
  do t = 0, use_threads - 1
    lo = (n * t) / use_threads + 1
    hi = (n * (t + 1)) / use_threads
    p = 0.0_c_double
    prod = 1.0_c_double
    do i = lo, hi
      prod = prod * c(i)
      p = c(i) * p + x(i)
    end do
    agg_A(t) = prod
    agg_B(t) = p
  end do
  !$omp end parallel do

  carry = 0.0_c_double
  carry_in(0) = carry
  do t = 1, use_threads - 1
    carry = agg_A(t - 1) * carry + agg_B(t - 1)
    carry_in(t) = carry
  end do

  !$omp parallel do schedule(static) default(shared) private(t, lo, hi, i, p)
  do t = 0, use_threads - 1
    lo = (n * t) / use_threads + 1
    hi = (n * (t + 1)) / use_threads
    p = carry_in(t)
    do i = lo, hi
      p = c(i) * p + x(i)
      y(i) = p
    end do
  end do
  !$omp end parallel do

  deallocate(agg_A, agg_B, carry_in)
end subroutine scan_affine_decay_fp64
