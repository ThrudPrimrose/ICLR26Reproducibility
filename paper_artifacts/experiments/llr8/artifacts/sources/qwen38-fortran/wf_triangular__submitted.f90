module wfmod
  use iso_c_binding
  implicit none
  integer, parameter :: B = 256
  integer, parameter :: NT = 4
contains
  subroutine impl(a, len_2d)
    implicit none
    real(c_double), intent(inout) :: a(*)
    integer(c_int), value, intent(in) :: len_2d
    integer(c_int64_t) :: n, m, nt, d, r, c, i0, i1, j0, j1, i, j, js, rlo, rhi
    n = int(len_2d, 8)
    if (n < 2) return
    m = n - 1
    nt = (m + B - 1) / B
    !$omp parallel num_threads(NT)
    do d = 0, 2*nt - 2
      rlo = max(int(0,8), d - nt + 1)
      rhi = min(d / 2, nt - 1)
      if (rlo > rhi) cycle
      !$omp do schedule(static)
      do r = rlo, rhi
        c = d - r
        i0 = r*B + 1; i1 = min((r+1)*B, m)
        j0 = c*B + 1; j1 = min((c+1)*B, m)
        do i = i0, i1
          js = max(j0, i)
          if (js > j1) cycle
          do j = js, j1
            a(i*n+j+1) = a(i*n+j+1) + a((i-1)*n+j+1) + a(i*n+j)
          end do
        end do
      end do
      !$omp end do
    end do
    !$omp end parallel
  end subroutine
end module wfmod
subroutine wf_triangular_fp64(a, len_2d) bind(C, name="wf_triangular_fp64")
  use wfmod
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  integer(c_int), value, intent(in) :: len_2d
  call impl(a, len_2d)
end subroutine
subroutine wf_triangular(a, len_2d) bind(C, name="wf_triangular")
  use wfmod
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  integer(c_int), value, intent(in) :: len_2d
  call impl(a, len_2d)
end subroutine
