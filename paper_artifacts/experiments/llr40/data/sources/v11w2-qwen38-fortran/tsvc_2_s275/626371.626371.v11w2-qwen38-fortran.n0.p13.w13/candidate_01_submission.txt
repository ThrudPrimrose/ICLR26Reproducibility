subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d) bind(c, name="tsvc_2_s275_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  use, intrinsic :: omp_lib, only: omp_get_num_threads
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in)    :: bb(*)
  real(c_double), intent(in)    :: cc(*)
  integer(c_int64_t), value, intent(in) :: len_2d
  integer(c_int64_t) :: n, i, j, lo, hi, k
  integer :: nt

  n = len_2d
  if (n < 2) return

  if (n <= 512) then
     ! small case: single thread, vectorizable row-major scan
     do j = 1, n - 1
        do i = 0, n - 1
           if (aa(1 + i) > 0.0d0) then
              aa(1 + i + j*n) = aa(1 + i + (j-1)*n) + &
                                bb(1 + i + j*n) * cc(1 + i + j*n)
           end if
        end do
     end do
     return
  end if

  !$omp parallel default(none) shared(aa,bb,cc,n) private(i,j,lo,hi,k,nt)
  nt = omp_get_num_threads()
  !$omp do schedule(static)
  do k = 1, nt
     lo = ((k-1)*n)/nt
     hi = (k*n)/nt - 1
     do j = 1, n - 1
        do i = lo, hi
           if (aa(1 + i) > 0.0d0) then
              aa(1 + i + j*n) = aa(1 + i + (j-1)*n) + &
                                bb(1 + i + j*n) * cc(1 + i + j*n)
           end if
        end do
     end do
  end do
  !$omp end do
  !$omp end parallel
end subroutine
