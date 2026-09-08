subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d) bind(C, name="tsvc_2_s275_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  integer(kind=c_int64_t), value :: len_2d
  real(kind=c_double), intent(inout), dimension(len_2d, len_2d) :: aa
  real(kind=c_double), intent(in),    dimension(len_2d, len_2d) :: bb
  real(kind=c_double), intent(in),    dimension(len_2d, len_2d) :: cc
  logical, dimension(:), allocatable :: pos
  integer(kind=c_int64_t) :: i, j, i0, i1, nt, base, rem, size_b
  integer(kind=c_int64_t) :: b

  allocate(pos(len_2d))
  !$omp parallel do
  do i = 1, len_2d
     pos(i) = aa(i,1) > 0.0d0
  end do

  !$omp parallel
  nt = omp_get_num_threads()
  base = len_2d / nt
  rem  = mod(len_2d, nt)
  !$omp do schedule(static)
  do b = 0, nt-1
     if (b < rem) then
        size_b = base + 1
        i0 = 1 + b*base + b
     else
        size_b = base
        i0 = 1 + b*base + rem
     end if
     i1 = i0 + size_b - 1
     do j = 2, len_2d
        do i = i0, i1
           if (pos(i)) aa(i,j) = aa(i,j-1) + bb(i,j)*cc(i,j)
        end do
     end do
  end do
  !$omp end parallel
  deallocate(pos)
end subroutine tsvc_2_s275_fp64
