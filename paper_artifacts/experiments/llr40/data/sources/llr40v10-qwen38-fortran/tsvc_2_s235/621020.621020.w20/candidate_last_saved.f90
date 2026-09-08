subroutine tsvc_2_s235_fp64(a, aa, b, bb, cc, LEN_2D) bind(c, name='tsvc_2_s235_fp64')
  use omp_lib
  implicit none
  real(kind=8), dimension(*) :: a, aa, b, bb, cc
  integer(kind=8), value :: LEN_2D
  integer(kind=8) :: N, i, j, i0, i1, chunk
  integer :: t, nt

  N = LEN_2D

  !$omp parallel do
  do i = 1, N
     a(i) = a(i) + b(i)*cc(i)
  end do

  !$omp parallel
  t = omp_get_thread_num()
  nt = omp_get_num_threads()
  chunk = (N + nt*8 - 1) / (nt*8) * 8
  i0 = t*chunk + 1
  if (i0 <= N) then
     i1 = min(i0 + chunk - 1, N)
     do j = 2, N
        do i = i0, i1
           aa((j-1)*N + i) = aa((j-2)*N + i) + bb((j-1)*N + i) * a(i)
        end do
     end do
  end if
  !$omp end parallel
end subroutine
