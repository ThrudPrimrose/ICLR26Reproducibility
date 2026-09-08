subroutine versioned_distance_update_fp64(a, b, c, len_1d, k, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, k
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in)    :: b(len_1d)
  real(c_double), intent(in)    :: c(len_1d)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer :: n, kk, i, r
  real(c_double), pointer, contiguous :: ws(:)

  n  = int(len_1d)
  kk = int(k)
  print "(A,I0,A,I0,A,I0)", "SIZES N=", n, " K=", kk, " WS=", workspace_size
  flush (6)
  if (kk <= 0 .or. n <= kk) return

  if (kk <= 8) then
    do i = kk + 1, min(2*kk, n)
      a(i) = 0.75d0 * a(i - kk) + b(i) * c(i)
    end do
    do i = 2*kk + 1, n
      a(i) = 0.75d0 * a(i - kk) + b(i) * c(i)
    end do
  else
    !$omp parallel do schedule(static)
    do r = 1, kk
      do i = kk + r, n, kk
        a(i) = 0.75d0 * a(i - kk) + b(i) * c(i)
      end do
    end do
  end if
end subroutine versioned_distance_update_fp64
